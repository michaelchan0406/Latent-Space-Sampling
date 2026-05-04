#!/usr/bin/env python3
"""
evaluate_distribution_similarity.py

Distribution similarity evaluation for generated medical questions.

This script produces a unified similarity cache with the following metrics:
  - MMD: Maximum Mean Discrepancy over text embeddings, lower is better.
  - FBD: Frechet distance over text embedding distributions, lower is better.
  - KL : KL divergence between ICD chapter-code distributions, lower is better.

The resulting cache is compatible with make_tables_and_figures.py.

Input manifest, CSV or JSONL:
  method,path,label_path
  MedQuAD+PubMedQA (GT),HUGGINGFACE_MIXED,labels/labeled_MedQuAD_PubMedQA_GT.csv
  M1 GPT-5.4,outputs/baselines/m1__gpt-5.4.jsonl,labels/labeled_M1_Baseline_GPT5.4.csv
  Ours-gpt-5.4-stratifiedL2,outputs/ours/step4_final_refined_gpt-5.4.jsonl,labels/labeled_ours_gpt5.4-stratifiedL2.csv

Notes:
  - The text path is used for MMD/FBD.
  - The label_path is used for KL over ICD chapter_code.
  - If label_path is absent, --label_dir plus built-in label filename inference can be used.
  - For reviewer-facing code, the default policy is to fail on missing label files rather
    than silently using dummy labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "This script requires torch and transformers. "
        "Install them with: pip install torch transformers"
    ) from exc

try:
    from scipy.linalg import sqrtm
except ImportError as exc:
    raise ImportError(
        "This script requires scipy for Frechet distance. Install with: pip install scipy"
    ) from exc

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEXT_COLUMN_CANDIDATES = [
    "rewritten",
    "qwen_rewritten_text",
    "text",
    "question",
    "Question",
    "medical_question",
    "prompt",
    "instruction",
]

LABEL_COLUMN_CANDIDATES = [
    "chapter_code",
    "icd_chapter",
    "chapter",
    "label",
    "icd10_chapter",
]

SIM_METRICS = ["MMD", "FBD", "KL"]

# Optional compatibility mapping for the method names used in earlier notebooks.
# Users can avoid relying on this by providing label_path in the manifest.
BUILTIN_LABEL_FILENAMES = {
    "MedQuAD+PubMedQA (GT)": "labeled_MedQuAD_PubMedQA_GT.csv",

    "M1 Qwen3.5-f": "labeled_M1_Baseline_Qwen3.5-flash.csv",
    "M1 GPT-5.4": "labeled_M1_Baseline_GPT5.4.csv",
    "M1 Gemini3-f": "labeled_M1_Baseline_Gemini3-flash.csv",

    "M2 Qwen3.5-f": "labeled_M4_Baseline_Qwen3.5-flash.csv",
    "M2 GPT-5.4": "labeled_M4_Baseline_GPT5.4.csv",
    "M2 Gemini3-f": "labeled_M4_Baseline_Gemini3-flash.csv",

    "M4 Qwen3.5-f": "labeled_M4_Baseline_Qwen3.5-flash.csv",
    "M4 GPT-5.4": "labeled_M4_Baseline_GPT5.4.csv",
    "M4 Gemini3-f": "labeled_M4_Baseline_Gemini3-flash.csv",

    "Self-instruct_gpt-5.4": "labeled_Self-instruct_gpt-5.4.csv",
    "WizardLM-gpt-5.4": "labeled_WizardLM-gpt-5.4.csv",
    "Med-Rag-gpt-5.4": "labeled_Med-Rag-gpt-5.4.csv",

    "Self-instruct_gemini3-f": "labeled_Self-instruct_gemini3-f.csv",
    "WizardLM-gemini3-f": "labeled_WizardLM-gemini3-f.csv",
    "Med-Rag-gemini3-f": "labeled_Med-Rag-gemini3-f.csv",

    "Self-instruct_qwen3.5-f": "labeled_Self-instruct_qwen3.5-f.csv",
    "WizardLM-qwen3.5-f": "labeled_WizardLM-qwen3.5-f.csv",
    "Med-Rag-qwen3.5-f": "labeled_Med-Rag-qwen3.5-f.csv",
    "Med-Rag_qwen3.5-f": "labeled_Med-Rag-qwen3.5-f.csv",

    "Ours-gpt-5.4-stratifiedL2": "labeled_ours_gpt5.4-stratifiedL2.csv",
    "Ours-gemini3f-stratifiedL2": "labeled_ours_gemini3f-stratifiedL2.csv",
    "Ours-qwen3.5f-stratifiedL2": "labeled_ours_qwen3.5-stratifiedL2.csv",

    "global_gmm_gpt5.4": "labeled_global_gmm_gpt5.4.csv",
    "global_gmm_gemini3-f": "labeled_global_gmm_gemini3-f.csv",
    "global_gmm_qwen3.5-f": "labeled_global_gmm_qwen3.5-f.csv",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    path: str
    label_path: Optional[str] = None


# ---------------------------------------------------------------------
# Logging / reproducibility
# ---------------------------------------------------------------------

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


# ---------------------------------------------------------------------
# Manifest and loading
# ---------------------------------------------------------------------

def load_manifest(path: Path) -> List[MethodSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        required = {"method", "path"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV manifest must contain columns {required}. "
                f"Found: {df.columns.tolist()}"
            )

        specs: List[MethodSpec] = []
        for _, row in df.iterrows():
            method = str(row["method"]).strip()
            source_path = str(row["path"]).strip()
            label_path = None
            if "label_path" in df.columns and not pd.isna(row.get("label_path")):
                label_path = str(row["label_path"]).strip()
            if method:
                specs.append(MethodSpec(method=method, path=source_path, label_path=label_path))
        return specs

    if path.suffix.lower() == ".jsonl":
        specs = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "method" not in obj or "path" not in obj:
                    raise ValueError(
                        f"JSONL manifest line {line_no} must contain method and path."
                    )
                specs.append(
                    MethodSpec(
                        method=str(obj["method"]).strip(),
                        path=str(obj["path"]).strip(),
                        label_path=(
                            str(obj["label_path"]).strip()
                            if obj.get("label_path")
                            else None
                        ),
                    )
                )
        return specs

    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def find_text_column(df: pd.DataFrame, source_path: str, method_name: str) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"No valid text column found for method={method_name}. "
        f"File={source_path}. Columns={df.columns.tolist()}"
    )


def find_label_column(df: pd.DataFrame, label_path: Path, method_name: str) -> str:
    for col in LABEL_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"No valid label column found for method={method_name}. "
        f"Label file={label_path}. Columns={df.columns.tolist()}"
    )


def load_huggingface_gt_texts() -> List[str]:
    if load_dataset is None:
        raise ImportError(
            "datasets is required to load HUGGINGFACE_MIXED. "
            "Install with: pip install datasets"
        )

    medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    medquad_texts = [
        x.get("Question", x.get("question", ""))
        for x in medquad
    ]

    pubmedqa = load_dataset(
        "pubmed_qa",
        "pqa_unlabeled",
        split="train",
        trust_remote_code=True,
    )
    pubmedqa_texts = [
        x.get("question", "")
        for x in pubmedqa
    ]

    all_texts = medquad_texts + pubmedqa_texts
    return [
        str(t).strip()
        for t in all_texts
        if isinstance(t, str) and len(t.strip()) > 10
    ]


def load_texts(
    method_name: str,
    source_path: str,
    max_pool: int,
    seed: int,
) -> List[str]:
    if source_path == "HUGGINGFACE_MIXED" or method_name.lower().endswith("(gt)"):
        all_texts = load_huggingface_gt_texts()
    else:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found for {method_name}: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".txt":
            all_texts = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            df = None
        else:
            raise ValueError(f"Unsupported text file type: {path}")

        if suffix != ".txt":
            assert df is not None
            text_col = find_text_column(df, str(path), method_name)
            all_texts = df[text_col].dropna().astype(str).tolist()

    cleaned = [
        t.strip()
        for t in all_texts
        if isinstance(t, str) and len(t.strip()) > 10
    ]

    if len(cleaned) > max_pool:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cleaned), size=max_pool, replace=False)
        cleaned = [cleaned[i] for i in idx.tolist()]

    return cleaned


def safe_filename_from_method(method_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method_name).strip("_")


def resolve_label_path(
    spec: MethodSpec,
    label_dir: Optional[Path],
) -> Optional[Path]:
    if spec.label_path:
        return Path(spec.label_path)

    if label_dir is None:
        return None

    if spec.method in BUILTIN_LABEL_FILENAMES:
        return label_dir / BUILTIN_LABEL_FILENAMES[spec.method]

    return label_dir / f"labeled_{safe_filename_from_method(spec.method)}.csv"


def make_dummy_labels(n: int) -> List[str]:
    return [f"U{i % 100:02d}" for i in range(n)]


def load_labels(
    spec: MethodSpec,
    text_count: int,
    label_dir: Optional[Path],
    missing_label_policy: str,
) -> List[str]:
    label_path = resolve_label_path(spec, label_dir)

    if label_path is None or not label_path.exists():
        msg = f"Missing label file for {spec.method}. Resolved path={label_path}"
        if missing_label_policy == "error":
            raise FileNotFoundError(msg)
        if missing_label_policy == "skip":
            logging.warning("%s; skipping method.", msg)
            return []
        if missing_label_policy == "dummy":
            logging.warning("%s; using dummy labels by request.", msg)
            return make_dummy_labels(text_count)
        raise ValueError(f"Unknown missing_label_policy: {missing_label_policy}")

    df = pd.read_csv(label_path)
    label_col = find_label_column(df, label_path, spec.method)
    labels = df[label_col].dropna().astype(str).str.strip().tolist()
    labels = [x for x in labels if x]

    if not labels:
        raise ValueError(f"No valid labels loaded for {spec.method}: {label_path}")

    return labels


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------

class TextEmbedder:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length

        logging.info("Loading embedding model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        raw_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = raw_model.encoder if hasattr(raw_model, "encoder") else raw_model
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embs: List[np.ndarray] = []

        with torch.no_grad():
            for i in progress(
                range(0, len(texts), batch_size),
                desc="Embedding batches",
                leave=False,
            ):
                batch = list(texts[i : i + batch_size])
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                out = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                mask = inputs["attention_mask"].unsqueeze(-1).type_as(
                    out.last_hidden_state
                )
                embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1e-9)

                embs = F.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


# ---------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------

def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> float:
    combined = np.vstack([x, y])
    if len(combined) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combined), size=max_points, replace=False)
        combined = combined[idx]

    dists = pairwise_sq_dists(combined, combined)
    upper = dists[np.triu_indices_from(dists, k=1)]
    upper = upper[upper > 1e-12]

    if len(upper) == 0:
        return 1.0

    median_sq_dist = float(np.median(upper))
    return 1.0 / (2.0 * median_sq_dist + 1e-12)


def compute_rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    k_xx = np.exp(-gamma * pairwise_sq_dists(x, x))
    k_yy = np.exp(-gamma * pairwise_sq_dists(y, y))
    k_xy = np.exp(-gamma * pairwise_sq_dists(x, y))

    mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return max(mmd2, 0.0)


def compute_frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)

    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    if cov_x.ndim == 0:
        cov_x = np.array([[float(cov_x)]])
    if cov_y.ndim == 0:
        cov_y = np.array([[float(cov_y)]])

    cov_x = cov_x + np.eye(cov_x.shape[0]) * eps
    cov_y = cov_y + np.eye(cov_y.shape[0]) * eps

    diff = mu_x - mu_y
    covmean = sqrtm(cov_x @ cov_y)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fbd = float(diff @ diff + np.trace(cov_x + cov_y - 2.0 * covmean))
    return max(fbd, 0.0)


def label_distribution(labels: Sequence[str], support: Sequence[str], eps: float) -> np.ndarray:
    counts = {label: 0 for label in support}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    vec = np.array([counts.get(label, 0) for label in support], dtype=float)
    vec = vec + eps
    vec = vec / vec.sum()
    return vec


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def subsample_aligned(
    texts: Sequence[str],
    labels: Sequence[str],
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str], np.ndarray]:
    effective_n = min(len(texts), len(labels))
    if effective_n < sample_size:
        raise ValueError(f"effective_n={effective_n} < sample_size={sample_size}")

    idx = rng.choice(effective_n, size=sample_size, replace=False)
    sub_texts = [texts[i] for i in idx.tolist()]
    sub_labels = [labels[i] for i in idx.tolist()]
    return sub_texts, sub_labels, idx


def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_val, std_val


def csv_metric_key(metric: str) -> str:
    return f"{metric} ↓"


def format_metric(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.6f} ± {std_val:.6f}"


# ---------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------

def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def maybe_backup_cache(cache_path: Path, backup_path: Optional[Path]) -> None:
    if backup_path is None or not cache_path.exists() or backup_path.exists():
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, backup_path)
    logging.info("Created cache backup: %s", backup_path)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate_similarity(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    manifest = load_manifest(Path(args.manifest))
    if not manifest:
        raise ValueError("Manifest is empty.")

    label_dir = Path(args.label_dir) if args.label_dir else None

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = TextEmbedder(
        model_name=args.embedding_model,
        device=device,
        max_length=args.max_length,
    )

    texts_by_method: Dict[str, List[str]] = {}
    labels_by_method: Dict[str, List[str]] = {}
    embeddings_by_method: Dict[str, np.ndarray] = {}

    logging.info("Loading texts, labels, and embeddings.")

    for spec in manifest:
        logging.info("Loading method=%s", spec.method)

        try:
            texts = load_texts(
                method_name=spec.method,
                source_path=spec.path,
                max_pool=args.pool_size,
                seed=args.seed,
            )
            labels = load_labels(
                spec=spec,
                text_count=len(texts),
                label_dir=label_dir,
                missing_label_policy=args.missing_label_policy,
            )
        except Exception as exc:
            logging.warning("Skipping %s due to loading error: %s", spec.method, repr(exc))
            continue

        effective_n = min(len(texts), len(labels))

        if effective_n < args.min_samples:
            logging.warning(
                "Skipping %s: effective_n=%d below min_samples=%d.",
                spec.method,
                effective_n,
                args.min_samples,
            )
            continue

        texts = texts[:effective_n]
        labels = labels[:effective_n]

        texts_by_method[spec.method] = texts
        labels_by_method[spec.method] = labels
        embeddings_by_method[spec.method] = embedder.encode(texts, batch_size=args.batch_size)

    if args.gt_name not in embeddings_by_method:
        raise RuntimeError(
            f"Ground-truth method `{args.gt_name}` was not loaded. "
            "It is required for distribution similarity evaluation."
        )

    gt_texts = texts_by_method[args.gt_name]
    gt_labels = labels_by_method[args.gt_name]
    gt_embeddings = embeddings_by_method[args.gt_name]

    cache_path = Path(args.cache_file)
    csv_path = Path(args.csv_file)

    maybe_backup_cache(cache_path, Path(args.backup_file) if args.backup_file else None)
    cache = load_cache(cache_path)

    all_label_support = sorted(
        set(gt_labels).union(
            label
            for method, labels in labels_by_method.items()
            for label in labels
        )
    )

    rng = np.random.default_rng(args.seed)

    for method_name in progress(embeddings_by_method.keys(), desc="Evaluating similarity"):
        if method_name == args.gt_name and args.skip_gt_similarity:
            logging.info("Skipping GT self-comparison by request.")
            continue

        texts = texts_by_method[method_name]
        labels = labels_by_method[method_name]
        embeddings = embeddings_by_method[method_name]

        effective_n = min(len(texts), len(labels), len(embeddings))
        if effective_n < args.sample_size:
            logging.warning(
                "Skipping %s: effective_n=%d < sample_size=%d.",
                method_name,
                effective_n,
                args.sample_size,
            )
            continue

        if (
            not args.force_recompute
            and method_name in cache
            and all(metric in cache[method_name].get("plot_data", {}) for metric in SIM_METRICS)
        ):
            logging.info("%s: similarity metrics already cached. Skipping.", method_name)
            continue

        logging.info("Computing similarity metrics for %s", method_name)

        mmd_scores: List[float] = []
        fbd_scores: List[float] = []
        kl_scores: List[float] = []

        for run_idx in range(args.runs):
            run_rng = np.random.default_rng(args.seed + run_idx)

            gt_idx = run_rng.choice(len(gt_embeddings), size=args.sample_size, replace=False)
            gen_idx = run_rng.choice(effective_n, size=args.sample_size, replace=False)

            gt_embs = gt_embeddings[gt_idx]
            gen_embs = embeddings[gen_idx]

            gt_sub_labels = [gt_labels[i] for i in gt_idx.tolist()]
            gen_sub_labels = [labels[i] for i in gen_idx.tolist()]

            gamma = (
                args.rbf_gamma
                if args.rbf_gamma > 0
                else median_heuristic_gamma(
                    gt_embs,
                    gen_embs,
                    max_points=args.median_heuristic_points,
                    seed=args.seed + run_idx,
                )
            )

            mmd_scores.append(compute_rbf_mmd2(gt_embs, gen_embs, gamma=gamma))
            fbd_scores.append(compute_frechet_distance(gt_embs, gen_embs))

            p_gt = label_distribution(gt_sub_labels, all_label_support, eps=args.kl_epsilon)
            p_gen = label_distribution(gen_sub_labels, all_label_support, eps=args.kl_epsilon)
            kl_scores.append(compute_kl_divergence(p_gt, p_gen))

        metric_values = {
            "MMD": mean_and_std(mmd_scores),
            "FBD": mean_and_std(fbd_scores),
            "KL": mean_and_std(kl_scores),
        }

        csv_data: Dict[str, Any] = {"Model / Method": method_name}
        plot_data: Dict[str, Any] = {"Model": method_name}

        for metric, (mean_val, std_val) in metric_values.items():
            csv_data[csv_metric_key(metric)] = format_metric(mean_val, std_val)
            plot_data[metric] = {
                "mean": mean_val,
                "err": std_val,
            }

        cache[method_name] = {
            "csv_data": csv_data,
            "plot_data": plot_data,
            "metadata": {
                "sample_size": args.sample_size,
                "runs": args.runs,
                "embedding_model": args.embedding_model,
                "kl_support_size": len(all_label_support),
                "gt_name": args.gt_name,
            },
        }

        save_cache(cache, cache_path)

    rows: List[Dict[str, Any]] = []
    display_cols = ["Model / Method"] + [csv_metric_key(metric) for metric in SIM_METRICS]

    for method_name, entry in cache.items():
        csv_data = entry.get("csv_data", {})
        row = {col: csv_data.get(col, "") for col in display_cols}
        row["Model / Method"] = csv_data.get("Model / Method", method_name)
        rows.append(row)

    df = pd.DataFrame(rows, columns=display_cols)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    logging.info("Saved similarity cache: %s", cache_path)
    logging.info("Saved similarity CSV: %s", csv_path)

    print("\n" + "=" * 100)
    print(
        f"DISTRIBUTION SIMILARITY "
        f"(N={args.sample_size}, {args.runs} runs, Mean ± Std)"
    )
    print("=" * 100)
    print(df.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distribution similarity evaluation for medical benchmark generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--gt_name", type=str, default="MedQuAD+PubMedQA (GT)")

    parser.add_argument("--cache_file", type=str, default="outputs/unified_benchmark_cache.json")
    parser.add_argument("--backup_file", type=str, default="outputs/unified_benchmark_cache.backup.json")
    parser.add_argument("--csv_file", type=str, default="outputs/distribution_similarity_results.csv")

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/gtr-t5-base")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--pool_size", type=int, default=2000)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--min_samples", type=int, default=1000)

    parser.add_argument(
        "--missing_label_policy",
        type=str,
        default="error",
        choices=["error", "skip", "dummy"],
        help="What to do when a label file is missing.",
    )

    parser.add_argument("--rbf_gamma", type=float, default=-1.0, help="Use <=0 for median heuristic.")
    parser.add_argument("--median_heuristic_points", type=int, default=800)
    parser.add_argument("--kl_epsilon", type=float, default=1e-8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--skip_gt_similarity", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    evaluate_similarity(args)


if __name__ == "__main__":
    main()