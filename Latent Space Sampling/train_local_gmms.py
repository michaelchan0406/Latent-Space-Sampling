#!/usr/bin/env python3
"""
Train ICD-10-based local Gaussian Mixture Models for medical question embeddings.

This script is the local GMM training step used before synthetic embedding
sampling. It takes ICD-10-labeled medical questions, maps ICD-10 chapters into
broader super-clusters, computes L2-normalized GTR-T5 embeddings, and trains one
GaussianMixture model per super-cluster.

Core logic preserved from the original notebook:
    1. Load ICD-10-labeled medical questions.
    2. Map ICD-10 chapter codes to super-clusters.
    3. Encode cluster-specific positive medical texts with gtr-t5-base.
    4. L2-normalize embeddings.
    5. Fit a local GMM for each super-cluster.
    6. Save each GMM as a joblib artifact.

Removed from the notebook:
    - FATE classifier training
    - non-medical negative samples
    - ResNet classifier definitions
    - tau threshold estimation
    - medical-vs-nonmedical boundary training

Required packages:
    pip install torch transformers scikit-learn numpy pandas tqdm joblib

Example:
    python train_local_gmms.py \
        --labeled-csv ./artifacts/icd10_labeled/labeled_MedQuAD_PubMedQA_GT.csv \
        --output-dir ./artifacts/local_gmms \
        --text-column text \
        --chapter-column chapter_code \
        --encoder-model sentence-transformers/gtr-t5-base \
        --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_ENCODER_MODEL = "sentence-transformers/gtr-t5-base"


CLUSTER_MAP: Dict[str, str] = {
    "A00-B99": "Onco_Infect",
    "C00-D48": "Onco_Infect",

    "F00-F99": "Neuro_Psych",
    "G00-G99": "Neuro_Psych",

    "E00-E90": "Internal_Med",
    "I00-I99": "Internal_Med",
    "J00-J99": "Internal_Med",
    "K00-K93": "Internal_Med",
    "D50-D89": "Internal_Med",

    "M00-M99": "Structural",
    "L00-L99": "Structural",
    "S00-T98": "Structural",

    "N00-N99": "Reproductive",
    "O00-O99": "Reproductive",
    "P00-P96": "Reproductive",

    "Q00-Q99": "Congenital_Sensory",
    "H00-H59": "Congenital_Sensory",
    "H60-H95": "Congenital_Sensory",

    "R00-R99": "General_Health",
    "Z00-Z99": "General_Health",

    # ICD-10 chapters that were not explicitly assigned in the original notebook
    # can be mapped to General_Health by fallback logic.
    "U00-U99": "General_Health",
    "V01-Y98": "General_Health",
}


@dataclass(frozen=True)
class Config:
    """Runtime configuration for local GMM training."""

    labeled_csv: Path
    output_dir: Path
    text_column: str
    chapter_column: str
    encoder_model: str
    seed: int
    device: str

    embedding_batch_size: int
    max_length: int
    min_cluster_size: int

    max_components: int
    samples_per_component: int
    covariance_type: str
    reg_covar: float
    max_iter: int
    n_init: int

    force: bool
    save_cluster_embeddings: bool


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device argument into a torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def sanitize_name(name: str) -> str:
    """Create a filesystem-safe cluster name."""
    allowed = []
    for char in name:
        if char.isalnum() or char in {"_", "-", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "cluster"


def load_labeled_questions(
    csv_path: Path,
    *,
    text_column: str,
    chapter_column: str,
) -> pd.DataFrame:
    """Load ICD-10-labeled questions from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Labeled CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in [text_column, chapter_column] if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[text_column, chapter_column]].rename(
        columns={
            text_column: "text",
            chapter_column: "chapter_code",
        }
    )

    df["text"] = df["text"].astype(str).str.strip()
    df["chapter_code"] = df["chapter_code"].astype(str).str.strip()

    df = df[
        (df["text"] != "")
        & (df["chapter_code"] != "")
        & (df["chapter_code"].str.lower() != "nan")
    ].copy()

    if df.empty:
        raise ValueError("No valid labeled rows found after cleaning.")

    df = df.drop_duplicates(subset=["text", "chapter_code"]).reset_index(drop=True)

    return df


def add_super_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ICD-10 chapter codes into broader medical super-clusters.

    Unmapped chapter codes fall back to General_Health, matching the notebook's
    fillna behavior.
    """
    out = df.copy()
    out["super_cluster"] = out["chapter_code"].map(CLUSTER_MAP).fillna("General_Health")
    return out


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token representations using the attention mask."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class GtrT5Embedder:
    """GTR-T5 encoder wrapper with mean pooling and L2 normalization."""

    def __init__(
        self,
        *,
        model_name: str,
        device: torch.device,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        print(f"[INFO] Loading encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        self.encoder = model.encoder if hasattr(model, "encoder") else model
        self.encoder.eval()

    def encode_l2(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        desc: str = "Encoding",
    ) -> np.ndarray:
        """Encode texts into L2-normalized embeddings."""
        if not texts:
            raise ValueError("No texts provided for embedding.")

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc=desc):
                batch = list(texts[start : start + batch_size])

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                embeddings = mean_pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )

                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().float().numpy())

        return np.vstack(all_embeddings).astype(np.float32)


def choose_n_components(
    n_samples: int,
    *,
    max_components: int,
    samples_per_component: int,
) -> int:
    """
    Choose GMM component count.

    This generalizes the notebook formula:
        n_comp = min(15, len(E_pos) // 20 + 1)

    Defaults reproduce the original behavior:
        max_components = 15
        samples_per_component = 20
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if samples_per_component <= 0:
        raise ValueError("samples_per_component must be positive.")

    return max(1, min(max_components, n_samples // samples_per_component + 1))


def train_gmm(
    embeddings: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    reg_covar: float,
    max_iter: int,
    n_init: int,
    seed: int,
) -> GaussianMixture:
    """Fit a GaussianMixture model on cluster embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {embeddings.shape}.")

    if embeddings.shape[0] < n_components:
        raise ValueError(
            f"n_components={n_components} cannot exceed number of samples={embeddings.shape[0]}."
        )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
    )

    gmm.fit(embeddings)
    return gmm


def save_metadata(
    *,
    output_dir: Path,
    metadata: Mapping[str, object],
) -> Path:
    """Save JSON metadata for reproducibility."""
    metadata_path = output_dir / "local_gmm_training_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path


def train_local_gmms(config: Config) -> pd.DataFrame:
    """Train one local GMM per ICD-10-derived super-cluster."""
    set_seed(config.seed)
    device = resolve_device(config.device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    gmm_dir = config.output_dir / "gmms"
    emb_dir = config.output_dir / "cluster_embeddings"
    gmm_dir.mkdir(parents=True, exist_ok=True)

    if config.save_cluster_embeddings:
        emb_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Local GMM training started.")
    print(f"[INFO] Labeled CSV: {config.labeled_csv}")
    print(f"[INFO] Output directory: {config.output_dir}")
    print(f"[INFO] Device: {device}")

    df = load_labeled_questions(
        config.labeled_csv,
        text_column=config.text_column,
        chapter_column=config.chapter_column,
    )
    df = add_super_cluster_labels(df)

    print(f"[INFO] Valid labeled questions: {len(df)}")
    print("[INFO] ICD-10 super-cluster distribution:")
    print(df["super_cluster"].value_counts().sort_index().to_string())

    embedder = GtrT5Embedder(
        model_name=config.encoder_model,
        device=device,
        max_length=config.max_length,
    )

    summary_records: List[Dict[str, object]] = []

    clusters = sorted(df["super_cluster"].dropna().unique().tolist())

    for cluster_name in clusters:
        safe_cluster = sanitize_name(cluster_name)
        gmm_path = gmm_dir / f"gmm_{safe_cluster}.joblib"
        emb_path = emb_dir / f"embeddings_{safe_cluster}.npy"

        cluster_texts = (
            df.loc[df["super_cluster"] == cluster_name, "text"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s != ""]
            .drop_duplicates()
            .tolist()
        )

        n_texts = len(cluster_texts)

        print("\n" + "=" * 72)
        print(f">>> Cluster: {cluster_name}")
        print("=" * 72)
        print(f"[INFO] Cluster texts: {n_texts}")

        if n_texts < config.min_cluster_size:
            print(
                f"[WARN] Skipping cluster '{cluster_name}': "
                f"{n_texts} samples < min_cluster_size={config.min_cluster_size}.",
                file=sys.stderr,
            )

            summary_records.append(
                {
                    "cluster": cluster_name,
                    "status": "skipped_too_small",
                    "n_texts": n_texts,
                    "n_components": 0,
                    "gmm_path": "",
                    "embedding_path": "",
                }
            )
            continue

        if gmm_path.exists() and not config.force:
            print(f"[INFO] GMM already exists; skipping: {gmm_path}")

            summary_records.append(
                {
                    "cluster": cluster_name,
                    "status": "exists",
                    "n_texts": n_texts,
                    "n_components": None,
                    "gmm_path": str(gmm_path),
                    "embedding_path": str(emb_path) if emb_path.exists() else "",
                }
            )
            continue

        print(f"[INFO] Encoding cluster texts for: {cluster_name}")
        embeddings = embedder.encode_l2(
            cluster_texts,
            batch_size=config.embedding_batch_size,
            desc=f"Encoding {cluster_name}",
        )

        if embeddings.shape[0] != n_texts:
            raise RuntimeError(
                f"Embedding count mismatch for cluster '{cluster_name}': "
                f"{embeddings.shape[0]} embeddings vs {n_texts} texts."
            )

        norms = np.linalg.norm(embeddings, axis=1)
        print(
            f"[INFO] Embedding shape: {embeddings.shape}; "
            f"L2 norm mean/std: {norms.mean():.6f}/{norms.std():.6f}"
        )

        if config.save_cluster_embeddings:
            np.save(emb_path, embeddings)
            print(f"[SAVE] Cluster embeddings: {emb_path}")

        n_components = choose_n_components(
            embeddings.shape[0],
            max_components=config.max_components,
            samples_per_component=config.samples_per_component,
        )

        print(
            f"[INFO] Fitting local GMM for {cluster_name}: "
            f"n_components={n_components}, covariance_type={config.covariance_type}"
        )

        gmm = train_gmm(
            embeddings,
            n_components=n_components,
            covariance_type=config.covariance_type,
            reg_covar=config.reg_covar,
            max_iter=config.max_iter,
            n_init=config.n_init,
            seed=config.seed,
        )

        joblib.dump(gmm, gmm_path)
        print(f"[SAVE] Local GMM: {gmm_path}")

        summary_records.append(
            {
                "cluster": cluster_name,
                "status": "trained",
                "n_texts": n_texts,
                "embedding_dim": int(embeddings.shape[1]),
                "n_components": int(n_components),
                "covariance_type": config.covariance_type,
                "reg_covar": float(config.reg_covar),
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
                "lower_bound": float(gmm.lower_bound_),
                "gmm_path": str(gmm_path),
                "embedding_path": str(emb_path) if config.save_cluster_embeddings else "",
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_path = config.output_dir / "local_gmm_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "cluster_map": CLUSTER_MAP,
        "n_total_labeled_questions": int(len(df)),
        "super_cluster_counts": {
            str(k): int(v)
            for k, v in df["super_cluster"].value_counts().sort_index().items()
        },
        "summary_csv": str(summary_path),
    }

    metadata_path = save_metadata(
        output_dir=config.output_dir,
        metadata=metadata,
    )

    print("\n" + "=" * 80)
    print(f"{'LOCAL GMM TRAINING SUMMARY':^80}")
    print("=" * 80)

    if summary_df.empty:
        print("[WARN] No clusters were processed.")
    else:
        print(summary_df.to_string(index=False))

    print("=" * 80)
    print(f"[SAVE] Summary CSV: {summary_path}")
    print(f"[SAVE] Metadata JSON: {metadata_path}")
    print("[DONE] Local ICD-10-based GMM training completed.")

    return summary_df


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ICD-10-based local GMMs on L2-normalized medical question embeddings."
    )

    parser.add_argument(
        "--labeled-csv",
        type=Path,
        required=True,
        help="CSV file containing ICD-10-labeled questions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where local GMM artifacts will be saved.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing question text. Default: text.",
    )
    parser.add_argument(
        "--chapter-column",
        type=str,
        default="chapter_code",
        help="Column containing ICD-10 chapter code. Default: chapter_code.",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default=DEFAULT_ENCODER_MODEL,
        help=f"Embedding encoder. Default: {DEFAULT_ENCODER_MODEL}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed. Default: 2026.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Batch size for embedding computation. Default: 256.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenizer sequence length. Default: 128.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="Minimum number of texts required to train a cluster GMM. Default: 50.",
    )

    parser.add_argument(
        "--max-components",
        type=int,
        default=15,
        help="Maximum number of GMM components per cluster. Default: 15.",
    )
    parser.add_argument(
        "--samples-per-component",
        type=int,
        default=20,
        help=(
            "Controls automatic component count: "
            "n_components = min(max_components, n_samples // samples_per_component + 1). "
            "Default: 20."
        ),
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="GMM covariance type. Default: full.",
    )
    parser.add_argument(
        "--reg-covar",
        type=float,
        default=1e-6,
        help="Non-negative regularization added to covariance diagonal. Default: 1e-6.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum EM iterations for GaussianMixture. Default: 100.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="Number of GMM initializations. Default: 1.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and overwrite existing GMM files.",
    )
    parser.add_argument(
        "--save-cluster-embeddings",
        action="store_true",
        help="Save per-cluster L2 embeddings as .npy files.",
    )

    args = parser.parse_args(argv)

    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive.")

    if args.max_length <= 0:
        raise ValueError("--max-length must be positive.")

    if args.min_cluster_size <= 0:
        raise ValueError("--min-cluster-size must be positive.")

    if args.max_components <= 0:
        raise ValueError("--max-components must be positive.")

    if args.samples_per_component <= 0:
        raise ValueError("--samples-per-component must be positive.")

    if args.reg_covar < 0:
        raise ValueError("--reg-covar must be non-negative.")

    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive.")

    if args.n_init <= 0:
        raise ValueError("--n-init must be positive.")

    return Config(
        labeled_csv=args.labeled_csv,
        output_dir=args.output_dir,
        text_column=args.text_column,
        chapter_column=args.chapter_column,
        encoder_model=args.encoder_model,
        seed=args.seed,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        max_length=args.max_length,
        min_cluster_size=args.min_cluster_size,
        max_components=args.max_components,
        samples_per_component=args.samples_per_component,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=args.n_init,
        force=args.force,
        save_cluster_embeddings=args.save_cluster_embeddings,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    config = parse_args(argv)
    train_local_gmms(config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Local GMM training stopped by user.", file=sys.stderr)
        raise SystemExit(130)