#!/usr/bin/env python3
"""
Reviewer-friendly ablation pipeline: No Gibberish Detector.

This script implements the "No Gibberish Detector" ablation.

Compared with the full pipeline, the intended experimental change is:
  - Full pipeline:
      ICD-10 proportional local GMM sampling
      + latent question-form classifier
      + vec2text decoding
      + gibberish filtering
      + LLM clinical rewriting

  - This ablation:
      ICD-10 proportional local GMM sampling
      + latent question-form classifier
      + vec2text decoding
      + NO gibberish filtering
      + LLM clinical rewriting directly from raw vec2text outputs

This ablation tests whether the text-based gibberish detector contributes to
the final quality of generated medical benchmark questions.

No private paths, names, institutions, or API keys are embedded in this file.
Supply API credentials through environment variables.

Example, full run:
  python ablation_no_gibberish.py \\
    --data-root ./data \\
    --output-dir ./outputs/ablation_no_gibberish \\
    --steps all

Example, fast run using a cached Step 2 file:
  python ablation_no_gibberish.py \\
    --data-root ./data \\
    --output-dir ./outputs/ablation_no_gibberish \\
    --steps rewrite \\
    --cached-step2-jsonl ./outputs/full_pipeline/step2_texts.jsonl

Expected default input layout:
  data/
    experts/
      gmms/
        gmm_<cluster_name>.joblib
    models/
      question_detector_gtr_t5_base_resnet_L2/
        resnet_qvstmt_gtr_t5_base_L2.pt
        standardize_mu_sg_and_threshold_L2.npz
    references/
      labeled_MedQuAD_PubMedQA_GT.csv

Outputs:
  output-dir/
    run_config.json
    step1_vectors_proportional.npy
    step1_vectors_proportional_meta.jsonl
    step1_sampling_quotas.json
    step2_texts.jsonl
    step3_skipped_gibberish_input.jsonl
    step4_final_refined_no_gibberish.jsonl
    step4_no_gibberish.done.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


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
}


REWRITE_SYSTEM_PROMPT = """You are an expert Medical Fact-Checker and Senior Physician.

Your input consists of raw text generated from a latent embedding space.
CRITICAL WARNING: The input may contain "Morphological Hallucinations"—words that sound medical but DO NOT exist.

Your Task:
1) Audit the input: Identify non-existent or nonsensical medical terms.
2) Correct & Rewrite: Transform the input into a grammatically perfect, clinically valid medical question.
3) Mapping: If a term is hallucinated, map it to the most likely intended real-world medical concept.
4) Incoherent Data: If the input is purely random gibberish that cannot be salvaged into a valid medical inquiry, set "rewritten" to null.

Output strictly a JSON object containing a single key "results", which is a list of objects exactly like this:
{
  "results": [
    {
      "idx": 1,
      "rewritten": "What is the clinical progression of glomerulonephritis?",
      "is_medical": true,
      "correction_made": true,
      "reason": "Corrected a hallucinated medical term to a clinically valid concept."
    }
  ]
}

Constraints:
- Only output English.
- Do NOT provide medical advice; only refine and validate the text.
- Rewritten questions must end with a question mark (?).
- If the original intent is clear despite hallucinated terminology, preserve that intent while fixing terminology.
"""


@dataclass(frozen=True)
class PipelinePaths:
    data_root: Path
    output_dir: Path

    gmm_dir: Path
    reference_csv: Path
    question_model_pt: Path
    question_stats_npz: Path

    step1_vectors: Path
    step1_meta: Path
    step1_quotas: Path

    step2_texts: Path
    step3_skipped_gibberish_input: Path

    step4_refined: Path
    step4_done: Path

    run_config: Path


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    target_total: int
    candidate_batch_size: int
    max_trials_min: int
    max_trials_multiplier: int
    normalize_samples: bool

    embedding_dim: int
    classifier_width: int
    classifier_blocks: int

    vec2text_corrector: str
    vec2text_batch_size: int
    vec2text_steps: int
    vec2text_beam: int

    rewrite_model: str
    rewrite_batch_size: int
    rewrite_temperature: float
    rewrite_timeout: float
    rewrite_max_retries: int
    sleep_between_retries: float

    ablation: str = "no_gibberish_detector"
    intended_difference_from_full_pipeline: str = (
        "Use the same ICD-10 proportional local GMM sampling, question-form "
        "classifier, and vec2text decoding as the full pipeline, but skip the "
        "text-based gibberish detector and rewrite raw vec2text outputs directly."
    )


class ResidualBlock(nn.Module):
    """Residual block used by the binary embedding classifier."""

    def __init__(self, dim: int, p: float = 0.20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class ResNetBin(nn.Module):
    """Binary ResNet classifier over embedding vectors."""

    def __init__(
        self,
        in_dim: int = 768,
        width: int = 512,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, p=0.2) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x))).squeeze(1)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-gibberish-detector ablation pipeline."
    )

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--gmm-dir",
        type=Path,
        default=None,
        help="Directory containing local GMM files named gmm_<cluster>.joblib. "
        "Default: <data-root>/experts/gmms",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="Reference CSV containing an ICD-10 chapter_code column. "
        "Default: <data-root>/references/labeled_MedQuAD_PubMedQA_GT.csv",
    )
    parser.add_argument(
        "--question-model-dir",
        type=Path,
        default=None,
        help="Directory containing the question-form classifier checkpoint and stats. "
        "Default: <data-root>/models/question_detector_gtr_t5_base_resnet_L2",
    )
    parser.add_argument("--question-model-pt", type=Path, default=None)
    parser.add_argument("--question-stats-npz", type=Path, default=None)

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "sample", "invert", "rewrite"],
        help="Pipeline steps to run. This ablation intentionally has no gibberish step.",
    )

    parser.add_argument(
        "--cached-step2-jsonl",
        type=Path,
        default=None,
        help="Optional existing step2_texts.jsonl to use for fast no-gibberish rewrite.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-total", type=int, default=21000)
    parser.add_argument("--candidate-batch-size", type=int, default=2000)
    parser.add_argument("--max-trials-min", type=int, default=200000)
    parser.add_argument("--max-trials-multiplier", type=int, default=100)
    parser.add_argument(
        "--no-normalize-samples",
        action="store_true",
        help="Disable L2 normalization after sampling from each local GMM.",
    )

    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--classifier-width", type=int, default=512)
    parser.add_argument("--classifier-blocks", type=int, default=4)

    parser.add_argument("--vec2text-corrector", type=str, default="gtr-base")
    parser.add_argument("--vec2text-batch-size", type=int, default=40)
    parser.add_argument("--vec2text-steps", type=int, default=40)
    parser.add_argument("--vec2text-beam", type=int, default=4)

    parser.add_argument("--rewrite-model", type=str, default="gpt-5.4")
    parser.add_argument("--rewrite-batch-size", type=int, default=15)
    parser.add_argument("--rewrite-temperature", type=float, default=0.0)
    parser.add_argument("--rewrite-timeout", type=float, default=120.0)
    parser.add_argument("--rewrite-max-retries", type=int, default=3)
    parser.add_argument("--sleep-between-retries", type=float, default=2.0)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base-url", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gmm_dir = (
        args.gmm_dir.expanduser().resolve()
        if args.gmm_dir is not None
        else data_root / "experts" / "gmms"
    )
    reference_csv = (
        args.reference_csv.expanduser().resolve()
        if args.reference_csv is not None
        else data_root / "references" / "labeled_MedQuAD_PubMedQA_GT.csv"
    )

    question_model_dir = (
        args.question_model_dir.expanduser().resolve()
        if args.question_model_dir is not None
        else data_root / "models" / "question_detector_gtr_t5_base_resnet_L2"
    )
    question_model_pt = (
        args.question_model_pt.expanduser().resolve()
        if args.question_model_pt is not None
        else question_model_dir / "resnet_qvstmt_gtr_t5_base_L2.pt"
    )
    question_stats_npz = (
        args.question_stats_npz.expanduser().resolve()
        if args.question_stats_npz is not None
        else question_model_dir / "standardize_mu_sg_and_threshold_L2.npz"
    )

    return PipelinePaths(
        data_root=data_root,
        output_dir=output_dir,
        gmm_dir=gmm_dir,
        reference_csv=reference_csv,
        question_model_pt=question_model_pt,
        question_stats_npz=question_stats_npz,
        step1_vectors=output_dir / "step1_vectors_proportional.npy",
        step1_meta=output_dir / "step1_vectors_proportional_meta.jsonl",
        step1_quotas=output_dir / "step1_sampling_quotas.json",
        step2_texts=output_dir / "step2_texts.jsonl",
        step3_skipped_gibberish_input=output_dir / "step3_skipped_gibberish_input.jsonl",
        step4_refined=output_dir / "step4_final_refined_no_gibberish.jsonl",
        step4_done=output_dir / "step4_no_gibberish.done.txt",
        run_config=output_dir / "run_config.json",
    )


def make_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        seed=args.seed,
        target_total=args.target_total,
        candidate_batch_size=args.candidate_batch_size,
        max_trials_min=args.max_trials_min,
        max_trials_multiplier=args.max_trials_multiplier,
        normalize_samples=not args.no_normalize_samples,
        embedding_dim=args.embedding_dim,
        classifier_width=args.classifier_width,
        classifier_blocks=args.classifier_blocks,
        vec2text_corrector=args.vec2text_corrector,
        vec2text_batch_size=args.vec2text_batch_size,
        vec2text_steps=args.vec2text_steps,
        vec2text_beam=args.vec2text_beam,
        rewrite_model=args.rewrite_model,
        rewrite_batch_size=args.rewrite_batch_size,
        rewrite_temperature=args.rewrite_temperature,
        rewrite_timeout=args.rewrite_timeout,
        rewrite_max_retries=args.rewrite_max_retries,
        sleep_between_retries=args.sleep_between_retries,
    )


def normalize_steps(raw_steps: Sequence[str]) -> List[str]:
    if "all" in raw_steps:
        return ["sample", "invert", "rewrite"]

    ordered = ["sample", "invert", "rewrite"]
    requested = set(raw_steps)
    return [step for step in ordered if step in requested]


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    require_file(path, "JSONL file")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc

    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def log_norm_stats(name: str, x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1)
    LOGGER.info(
        "%s norm stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
        name,
        float(norms.mean()),
        float(norms.std()),
        float(norms.min()),
        float(norms.max()),
    )


def find_gmm_clusters(gmm_dir: Path) -> List[str]:
    require_dir(gmm_dir, "GMM directory")

    clusters = []
    for path in gmm_dir.glob("gmm_*.joblib"):
        stem = path.stem
        if stem.startswith("gmm_"):
            clusters.append(stem.split("_", 1)[1])

    clusters = sorted(set(clusters))
    if not clusters:
        raise FileNotFoundError(f"No gmm_<cluster>.joblib files found in {gmm_dir}")

    return clusters


def load_reference_distribution(reference_csv: Path) -> pd.Series:
    require_file(reference_csv, "reference CSV")

    ref_df = pd.read_csv(reference_csv)
    if "chapter_code" not in ref_df.columns:
        raise ValueError(
            f"Reference CSV must contain a 'chapter_code' column. "
            f"Found columns: {list(ref_df.columns)}"
        )

    ref_df = ref_df.copy()
    ref_df["super_cluster"] = ref_df["chapter_code"].map(CLUSTER_MAP).fillna(
        "General_Health"
    )

    counts = ref_df["super_cluster"].value_counts()
    if counts.sum() == 0:
        raise ValueError(f"Reference distribution is empty: {reference_csv}")

    return counts / counts.sum()


def allocate_largest_remainder(
    proportions: pd.Series,
    target_total: int,
) -> Dict[str, int]:
    if target_total <= 0:
        raise ValueError("--target-total must be positive.")

    proportions = proportions / proportions.sum()
    raw_quotas = proportions * target_total
    quotas = np.floor(raw_quotas).astype(int)

    remainder = int(target_total - quotas.sum())
    if remainder > 0:
        fractional = raw_quotas - np.floor(raw_quotas)
        for cluster in fractional.sort_values(ascending=False).index[:remainder]:
            quotas[cluster] += 1

    return {str(cluster): int(quota) for cluster, quota in quotas.items()}


def save_sampling_quotas(
    path: Path,
    valid_clusters: Sequence[str],
    valid_props: pd.Series,
    quotas: Dict[str, int],
    missing_in_gmm: Sequence[str],
) -> None:
    payload = {
        "ablation": "no_gibberish_detector",
        "sampling": "ICD-10 proportional stratified local GMM sampling",
        "question_classifier": "enabled",
        "gibberish_detector": "disabled",
        "valid_clusters": list(valid_clusters),
        "missing_reference_clusters_without_gmm": list(missing_in_gmm),
        "quotas": {
            cluster: {
                "reference_proportion": float(valid_props[cluster]),
                "target_n": int(quotas[cluster]),
            }
            for cluster in valid_clusters
        },
        "target_total": int(sum(quotas.values())),
    }

    save_json(payload, path)


def load_question_classifier(
    checkpoint_path: Path,
    stats_path: Path,
    device: torch.device,
    in_dim: int,
    width: int,
    num_blocks: int,
) -> Tuple[ResNetBin, np.ndarray, np.ndarray, float]:
    require_file(checkpoint_path, "question classifier checkpoint")
    require_file(stats_path, "question classifier stats")

    model = ResNetBin(in_dim=in_dim, width=width, num_blocks=num_blocks)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    stats = np.load(stats_path)

    if "mu" not in stats.files or "sg" not in stats.files:
        raise KeyError(f"Expected 'mu' and 'sg' arrays in {stats_path}")

    mu = stats["mu"].squeeze().astype(np.float32)
    sg = stats["sg"].squeeze().astype(np.float32)

    if "best_th" in stats.files:
        threshold = float(stats["best_th"][0])
    elif "threshold" in stats.files:
        threshold = float(stats["threshold"][0])
    else:
        raise KeyError(f"Expected 'best_th' or 'threshold' in {stats_path}")

    if mu.shape[0] != in_dim or sg.shape[0] != in_dim:
        raise ValueError(
            f"Classifier stats dimension mismatch: mu={mu.shape}, sg={sg.shape}, "
            f"expected={in_dim}"
        )

    sg = np.where(np.abs(sg) < 1e-12, 1.0, sg).astype(np.float32)

    return model, mu, sg, threshold


def predict_question_probs(
    model: nn.Module,
    mu: np.ndarray,
    sg: np.ndarray,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = (x[start : start + batch_size].astype(np.float32) - mu) / sg
            xb_tensor = torch.tensor(xb, dtype=torch.float32, device=device)
            logits = model(xb_tensor).detach().cpu().numpy()
            probs.append(1.0 / (1.0 + np.exp(-logits)))

    return np.concatenate(probs, axis=0).astype(np.float32)


def step1_stratified_sampling_with_question_filter(
    paths: PipelinePaths,
    config: PipelineConfig,
    device: torch.device,
    overwrite: bool,
) -> np.ndarray:
    """
    Step 1: ICD-10 proportional local GMM sampling with question-form filtering.

    This matches the full pipeline up to Step 1.
    The ablation does not alter sampling or question filtering.
    """
    LOGGER.info("Step 1: ICD-10 proportional local GMM sampling with question classifier")

    if paths.step1_vectors.exists() and paths.step1_meta.exists() and not overwrite:
        x_cached = np.load(paths.step1_vectors).astype(np.float32)
        if config.normalize_samples:
            x_cached = l2_normalize(x_cached)
        LOGGER.info(
            "Loaded cached Step 1 vectors: %s shape=%s",
            paths.step1_vectors,
            x_cached.shape,
        )
        log_norm_stats("Cached Step 1 vectors", x_cached)
        return x_cached

    if overwrite:
        for path in (paths.step1_vectors, paths.step1_meta, paths.step1_quotas):
            if path.exists():
                path.unlink()

    ref_props = load_reference_distribution(paths.reference_csv)
    gmm_clusters = find_gmm_clusters(paths.gmm_dir)

    valid_clusters = [cluster for cluster in gmm_clusters if cluster in ref_props.index]
    if not valid_clusters:
        raise ValueError(
            "No overlapping clusters found between local GMM files and reference distribution."
        )

    missing_in_gmm = sorted(set(ref_props.index) - set(valid_clusters))
    if missing_in_gmm:
        LOGGER.warning(
            "Reference clusters without matching local GMM will be ignored: %s",
            missing_in_gmm,
        )

    valid_props = ref_props.loc[valid_clusters]
    valid_props = valid_props / valid_props.sum()

    quotas = allocate_largest_remainder(valid_props, config.target_total)
    save_sampling_quotas(
        path=paths.step1_quotas,
        valid_clusters=valid_clusters,
        valid_props=valid_props,
        quotas=quotas,
        missing_in_gmm=missing_in_gmm,
    )

    q_model, q_mu, q_sg, q_th = load_question_classifier(
        checkpoint_path=paths.question_model_pt,
        stats_path=paths.question_stats_npz,
        device=device,
        in_dim=config.embedding_dim,
        width=config.classifier_width,
        num_blocks=config.classifier_blocks,
    )

    LOGGER.info("Question classifier threshold: %.6f", q_th)
    LOGGER.info("Sampling quotas:")
    for cluster in valid_clusters:
        LOGGER.info(
            "  %-25s reference=%6.2f%% target_n=%d",
            cluster,
            valid_props[cluster] * 100.0,
            quotas[cluster],
        )

    final_vectors: List[np.ndarray] = []
    final_meta: List[Dict[str, Any]] = []

    for cluster_name in valid_clusters:
        target_n = quotas[cluster_name]
        if target_n <= 0:
            continue

        gmm_path = paths.gmm_dir / f"gmm_{cluster_name}.joblib"
        require_file(gmm_path, f"local GMM for cluster {cluster_name}")

        LOGGER.info("Sampling cluster %s from %s", cluster_name, gmm_path)
        gmm = joblib.load(gmm_path)

        accepted = 0
        trials = 0
        max_trials = max(
            config.max_trials_min,
            target_n * config.max_trials_multiplier,
        )

        pbar = tqdm(total=target_n, desc=f"Cluster {cluster_name}")

        while accepted < target_n and trials < max_trials:
            x_candidate, component_ids = gmm.sample(config.candidate_batch_size)
            x_candidate = x_candidate.astype(np.float32)

            if config.normalize_samples:
                x_candidate = l2_normalize(x_candidate)

            trials += x_candidate.shape[0]

            q_probs = predict_question_probs(
                model=q_model,
                mu=q_mu,
                sg=q_sg,
                x=x_candidate,
                device=device,
            )
            keep_mask = q_probs >= q_th

            x_keep = x_candidate[keep_mask]
            comp_keep = component_ids[keep_mask]
            q_keep = q_probs[keep_mask]

            if x_keep.shape[0] == 0:
                continue

            for vector, component_id, q_score in zip(x_keep, comp_keep, q_keep):
                if accepted >= target_n:
                    break

                final_vectors.append(vector.astype(np.float32))
                final_meta.append(
                    {
                        "idx": len(final_meta),
                        "source": "icd10_local_gmm_question_filtered_no_gibberish_ablation",
                        "source_cluster": cluster_name,
                        "target_quota": int(target_n),
                        "gmm_component": int(component_id),
                        "q_prob": float(q_score),
                    }
                )
                accepted += 1
                pbar.update(1)

        pbar.close()

        if accepted < target_n:
            LOGGER.warning(
                "Cluster %s generated only %d/%d vectors after %d sampled candidates.",
                cluster_name,
                accepted,
                target_n,
                trials,
            )

        del gmm
        gc.collect()

    if not final_vectors:
        raise RuntimeError(
            "No vectors were generated. Check local GMM paths, classifier files, and thresholds."
        )

    x_final = np.vstack(final_vectors).astype(np.float32)
    if config.normalize_samples:
        x_final = l2_normalize(x_final)

    np.save(paths.step1_vectors, x_final)
    append_jsonl(paths.step1_meta, final_meta)

    LOGGER.info("Saved Step 1 vectors: %s shape=%s", paths.step1_vectors, x_final.shape)
    LOGGER.info("Saved Step 1 metadata: %s", paths.step1_meta)
    log_norm_stats("Step 1 saved vectors", x_final)

    return x_final


def done_indices_from_jsonl(path: Path) -> Set[int]:
    done: Set[int] = set()

    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["idx"]))
            except Exception:
                continue

    return done


def step2_vec2text_inversion(
    paths: PipelinePaths,
    config: PipelineConfig,
    device: torch.device,
    overwrite: bool,
) -> None:
    """Step 2: Decode accepted vectors into raw text using vec2text."""
    LOGGER.info("Step 2: vec2text inversion")

    require_file(paths.step1_vectors, "Step 1 vector file")

    if overwrite and paths.step2_texts.exists():
        paths.step2_texts.unlink()

    x_target = np.load(paths.step1_vectors).astype(np.float32)
    if config.normalize_samples:
        x_target = l2_normalize(x_target)

    if x_target.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {x_target.shape[1]}, "
            f"expected {config.embedding_dim}"
        )

    log_norm_stats("Step 2 vec2text input", x_target)

    done = done_indices_from_jsonl(paths.step2_texts)
    if len(done) >= x_target.shape[0]:
        LOGGER.info("Step 2 already complete: %s", paths.step2_texts)
        return

    try:
        import vec2text  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "vec2text is required for Step 2. Install it or skip the 'invert' step."
        ) from exc

    corrector = vec2text.load_pretrained_corrector(config.vec2text_corrector)

    with torch.no_grad():
        for start in tqdm(
            range(0, x_target.shape[0], config.vec2text_batch_size),
            desc="vec2text inversion",
        ):
            end = min(start + config.vec2text_batch_size, x_target.shape[0])
            indices = [idx for idx in range(start, end) if idx not in done]

            if not indices:
                continue

            emb_batch = torch.from_numpy(x_target[indices]).to(device)

            decoded = vec2text.invert_embeddings(
                embeddings=emb_batch,
                corrector=corrector,
                num_steps=config.vec2text_steps,
                sequence_beam_width=config.vec2text_beam,
            )

            records = [
                {
                    "idx": int(idx),
                    "text": str(text),
                }
                for idx, text in zip(indices, decoded)
            ]
            append_jsonl(paths.step2_texts, records)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    LOGGER.info("Saved Step 2 decoded texts: %s", paths.step2_texts)


def prepare_no_gibberish_rewrite_input(
    paths: PipelinePaths,
    cached_step2_jsonl: Optional[Path],
    overwrite: bool,
) -> Path:
    """
    Prepare the rewrite input for the no-gibberish ablation.

    This intentionally does not run any gibberish detector. It either:
      - uses a user-provided cached Step 2 JSONL file, or
      - uses this run's output-dir/step2_texts.jsonl.

    A copy is written as step3_skipped_gibberish_input.jsonl to make the skipped
    step explicit in the output directory.
    """
    source = (
        cached_step2_jsonl.expanduser().resolve()
        if cached_step2_jsonl is not None
        else paths.step2_texts
    )

    require_file(source, "Step 2 vec2text JSONL input")

    if paths.step3_skipped_gibberish_input.exists() and not overwrite:
        LOGGER.info(
            "No-gibberish rewrite input already exists: %s",
            paths.step3_skipped_gibberish_input,
        )
        return paths.step3_skipped_gibberish_input

    if paths.step3_skipped_gibberish_input.exists() and overwrite:
        paths.step3_skipped_gibberish_input.unlink()

    shutil.copyfile(source, paths.step3_skipped_gibberish_input)

    LOGGER.info("Gibberish detector skipped by design.")
    LOGGER.info("Copied raw Step 2 texts from %s", source)
    LOGGER.info("Rewrite input saved as %s", paths.step3_skipped_gibberish_input)

    return paths.step3_skipped_gibberish_input


def make_openai_client(
    api_key_env: str,
    base_url: Optional[str],
    timeout: float,
) -> Any:
    if OpenAI is None:
        raise ImportError(
            "The openai package is required for rewrite. Install it or skip 'rewrite'."
        )

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set environment variable {api_key_env} before running rewrite."
        )

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
    }

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object.")

    return obj


def load_done_ids(done_path: Path, output_path: Path) -> Set[int]:
    done: Set[int] = set()

    if done_path.exists():
        with done_path.open("r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if not value:
                    continue
                try:
                    done.add(int(value))
                except Exception:
                    continue

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if "idx" in row:
                        done.add(int(row["idx"]))
                except Exception:
                    continue

    return done


def normalize_rewrite_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "idx" not in raw:
        raise ValueError(f"Missing idx in rewrite result: {raw}")

    rewritten = raw.get("rewritten")
    if rewritten is not None:
        rewritten = str(rewritten).strip()
        if rewritten.lower() in {"", "null", "none"}:
            rewritten = None

    return {
        "idx": int(raw["idx"]),
        "rewritten": rewritten,
        "is_medical": bool(raw.get("is_medical", rewritten is not None)),
        "correction_made": bool(raw.get("correction_made", False)),
        "reason": str(raw.get("reason", "")),
    }


def call_rewrite_batch(
    client: Any,
    items: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_retries: int,
    sleep_between_retries: float,
) -> List[Dict[str, Any]]:
    batch_idx = [int(item["idx"]) for item in items]
    payload = {
        "items": [
            {
                "idx": int(item["idx"]),
                "text": str(item.get("text", "")),
            }
            for item in items
        ]
    }

    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }

    for attempt in range(1, max_retries + 1):
        try:
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as json_mode_error:
                LOGGER.warning(
                    "JSON response_format failed; falling back without it: %s",
                    json_mode_error,
                )
                response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            obj = extract_json_object(content)

            results = obj.get("results", [])
            if not isinstance(results, list):
                raise ValueError("Bad rewrite JSON schema: 'results' is not a list.")

            normalized = [
                normalize_rewrite_result(result)
                for result in results
                if isinstance(result, dict)
            ]

            expected_idx = set(batch_idx)
            got_idx = {int(result["idx"]) for result in normalized}

            missing = expected_idx - got_idx
            extra = got_idx - expected_idx

            if missing or extra:
                raise ValueError(
                    f"Model result idx mismatch. "
                    f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
                )

            return normalized

        except Exception as exc:
            if attempt == max_retries:
                raise

            LOGGER.warning(
                "Rewrite batch failed on attempt %d/%d: %s",
                attempt,
                max_retries,
                exc,
            )
            time.sleep(sleep_between_retries)

    raise RuntimeError("Unreachable rewrite retry state.")


def step4_llm_rewrite_without_gibberish(
    paths: PipelinePaths,
    config: PipelineConfig,
    api_key_env: str,
    base_url: Optional[str],
    cached_step2_jsonl: Optional[Path],
    overwrite: bool,
) -> None:
    """Step 4: LLM rewriting directly from raw vec2text output."""
    LOGGER.info("Step 4: LLM rewrite WITHOUT gibberish filtering")

    rewrite_input = prepare_no_gibberish_rewrite_input(
        paths=paths,
        cached_step2_jsonl=cached_step2_jsonl,
        overwrite=overwrite,
    )

    if overwrite:
        for path in (paths.step4_refined, paths.step4_done):
            if path.exists():
                path.unlink()

    client = make_openai_client(
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=config.rewrite_timeout,
    )

    done_ids = load_done_ids(paths.step4_done, paths.step4_refined)

    total = count_jsonl(rewrite_input)
    LOGGER.info("Rewrite input: %s", rewrite_input)
    LOGGER.info("Rewrite output: %s", paths.step4_refined)
    LOGGER.info("Already done: %d", len(done_ids))
    LOGGER.info("Rewrite model: %s", config.rewrite_model)

    batch_in: List[Dict[str, Any]] = []
    batch_raw: List[Dict[str, Any]] = []

    output_mode = "a" if paths.step4_refined.exists() else "w"

    with rewrite_input.open("r", encoding="utf-8") as fin, \
        paths.step4_refined.open(output_mode, encoding="utf-8") as fout, \
        paths.step4_done.open("a", encoding="utf-8") as fdone:

        for line in tqdm(fin, total=total, desc="LLM rewrite, no gibberish"):
            if not line.strip():
                continue

            raw_record = json.loads(line)
            idx = int(raw_record["idx"])

            if idx in done_ids:
                continue

            text = str(raw_record.get("text", "")).strip()
            if not text:
                fdone.write(str(idx) + "\n")
                done_ids.add(idx)
                continue

            batch_in.append({"idx": idx, "text": text})
            batch_raw.append(raw_record)

            if len(batch_in) >= config.rewrite_batch_size:
                results = call_rewrite_batch(
                    client=client,
                    items=batch_in,
                    model=config.rewrite_model,
                    temperature=config.rewrite_temperature,
                    max_retries=config.rewrite_max_retries,
                    sleep_between_retries=config.sleep_between_retries,
                )

                result_map = {int(result["idx"]): result for result in results}

                for raw in batch_raw:
                    row_idx = int(raw["idx"])
                    result = result_map[row_idx]

                    out = {
                        "idx": row_idx,
                        "rewritten": result.get("rewritten"),
                        "is_medical": bool(result.get("is_medical", False)),
                        "correction_made": bool(result.get("correction_made", False)),
                        "reason": result.get("reason", ""),
                    }

                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fdone.write(str(row_idx) + "\n")
                    done_ids.add(row_idx)

                fout.flush()
                fdone.flush()
                batch_in.clear()
                batch_raw.clear()

        if batch_in:
            results = call_rewrite_batch(
                client=client,
                items=batch_in,
                model=config.rewrite_model,
                temperature=config.rewrite_temperature,
                max_retries=config.rewrite_max_retries,
                sleep_between_retries=config.sleep_between_retries,
            )

            result_map = {int(result["idx"]): result for result in results}

            for raw in batch_raw:
                row_idx = int(raw["idx"])
                result = result_map[row_idx]

                out = {
                    "idx": row_idx,
                    "rewritten": result.get("rewritten"),
                    "is_medical": bool(result.get("is_medical", False)),
                    "correction_made": bool(result.get("correction_made", False)),
                    "reason": result.get("reason", ""),
                }

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fdone.write(str(row_idx) + "\n")
                done_ids.add(row_idx)

            fout.flush()
            fdone.flush()

    LOGGER.info("Saved no-gibberish rewritten questions: %s", paths.step4_refined)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    seed_everything(args.seed)

    paths = resolve_paths(args)
    config = make_config(args)
    device = get_device(args.device)
    steps = normalize_steps(args.steps)

    save_json(
        {
            "paths": {key: str(value) for key, value in asdict(paths).items()},
            "config": asdict(config),
            "steps": steps,
            "device": str(device),
            "cached_step2_jsonl": str(args.cached_step2_jsonl) if args.cached_step2_jsonl else None,
        },
        paths.run_config,
    )

    LOGGER.info("Ablation: No Gibberish Detector")
    LOGGER.info("Output directory: %s", paths.output_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Steps: %s", steps)

    if "sample" in steps:
        step1_stratified_sampling_with_question_filter(
            paths=paths,
            config=config,
            device=device,
            overwrite=args.overwrite,
        )

    if "invert" in steps:
        step2_vec2text_inversion(
            paths=paths,
            config=config,
            device=device,
            overwrite=args.overwrite,
        )

    if "rewrite" in steps:
        step4_llm_rewrite_without_gibberish(
            paths=paths,
            config=config,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            cached_step2_jsonl=args.cached_step2_jsonl,
            overwrite=args.overwrite,
        )

    LOGGER.info(
        "No-gibberish-detector ablation completed. Results saved in %s",
        paths.output_dir,
    )


if __name__ == "__main__":
    main()