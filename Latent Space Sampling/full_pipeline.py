#!/usr/bin/env python3
"""
Reviewer-friendly full pipeline for synthetic medical benchmark question generation.

This script implements the anonymized full pipeline used for generating synthetic
medical benchmark questions from local ICD-10-derived embedding-space experts.

Pipeline:
  1. ICD-10-based proportional local GMM sampling.
  2. Question-form filtering using a trained binary classifier.
  3. vec2text decoding from embeddings to raw text.
  4. Text-based gibberish detection and filtering.
  5. LLM-based clinical rewriting / validation.

Important:
  - This is the full pipeline after removal of the previous FATE classifier branch.
  - No local identity information, private paths, institution names, or API keys are
    embedded in this file.
  - API credentials must be supplied through environment variables.
  - All major hard-coded parameters are exposed through CLI arguments.

Example:
  python full_pipeline.py \
    --data-root ./data \
    --output-dir ./outputs/full_pipeline \
    --target-total 21000 \
    --steps all

Example with LLM rewriting:
  export OPENAI_API_KEY="YOUR_API_KEY"
  python full_pipeline.py \
    --data-root ./data \
    --output-dir ./outputs/full_pipeline \
    --steps refine \
    --rewrite-model gpt-5.4

Expected directory structure by default:
  data/
    experts/
      gmms/
        gmm_<cluster_name>.joblib
    models/
      question_detector/
        resnet_qvstmt_gtr_t5_base_L2.pt
        standardize_mu_sg_and_threshold_L2.npz
    references/
      labeled_MedQuAD_PubMedQA_GT.csv

Outputs:
  output-dir/
    step1_vectors_proportional.npy
    step1_vectors_proportional_meta.jsonl
    step1_sampling_quotas.json
    step2_texts.jsonl
    step3_clean_texts.jsonl
    step4_final_refined.jsonl
    step4_refined.done.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import pipeline

try:
    from openai import OpenAI
except ImportError:  # Allows sampling/decoding/filtering to run without OpenAI installed.
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


SYSTEM_PROMPT = """You are an expert Medical Fact-Checker and Senior Physician.

Your input consists of raw text generated from a latent embedding space.
CRITICAL WARNING: The input may contain "Morphological Hallucinations"—words that sound medical but DO NOT exist, for example fabricated disease or anatomy names.

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
    """Resolved input/output paths for the pipeline."""

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
    step3_clean_texts: Path
    step4_refined: Path
    step4_done: Path


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
    """Binary classifier over embedding vectors."""

    def __init__(self, in_dim: int = 768, width: int = 512, num_blocks: int = 4) -> None:
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


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full synthetic medical benchmark question generation pipeline."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing anonymized data, experts, models, and references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where pipeline outputs will be written.",
    )

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
        "Default: <data-root>/models/question_detector",
    )
    parser.add_argument(
        "--question-model-pt",
        type=Path,
        default=None,
        help="Question-form classifier checkpoint. "
        "Default: <question-model-dir>/resnet_qvstmt_gtr_t5_base_L2.pt",
    )
    parser.add_argument(
        "--question-stats-npz",
        type=Path,
        default=None,
        help="Question-form classifier standardization stats and threshold. "
        "Default: <question-model-dir>/standardize_mu_sg_and_threshold_L2.npz",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "sample", "invert", "gibberish", "refine"],
        help="Pipeline steps to run.",
    )

    parser.add_argument(
        "--target-total",
        type=int,
        default=21000,
        help="Total number of vectors to sample across valid ICD-10 super-clusters.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=2000,
        help="Number of candidate embeddings sampled from each GMM per trial.",
    )
    parser.add_argument(
        "--max-trials-min",
        type=int,
        default=200000,
        help="Minimum maximum number of sampled candidates allowed per cluster.",
    )
    parser.add_argument(
        "--max-trials-multiplier",
        type=int,
        default=100,
        help="Cluster-specific max trials are max(max-trials-min, target_n * multiplier).",
    )
    parser.add_argument(
        "--no-normalize-samples",
        action="store_true",
        help="Disable L2 normalization of sampled GMM vectors.",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension expected by the question-form classifier.",
    )
    parser.add_argument(
        "--classifier-width",
        type=int,
        default=512,
        help="Hidden width of the ResNet binary classifier.",
    )
    parser.add_argument(
        "--classifier-blocks",
        type=int,
        default=4,
        help="Number of residual blocks in the ResNet binary classifier.",
    )

    parser.add_argument(
        "--vec2text-corrector",
        type=str,
        default="gtr-base",
        help="Name of the pretrained vec2text corrector.",
    )
    parser.add_argument(
        "--vec2text-batch-size",
        type=int,
        default=40,
        help="Batch size for vec2text inversion.",
    )
    parser.add_argument(
        "--vec2text-num-steps",
        type=int,
        default=40,
        help="Number of optimization/correction steps for vec2text inversion.",
    )

    parser.add_argument(
        "--disable-gibberish-filter",
        action="store_true",
        help="Skip gibberish filtering and copy step2_texts.jsonl to step3_clean_texts.jsonl.",
    )
    parser.add_argument(
        "--gibberish-model",
        type=str,
        default="madhurjindal/autonlp-Gibberish-Detector-492513457",
        help="Hugging Face text-classification model used for gibberish detection.",
    )
    parser.add_argument(
        "--gibberish-batch-size",
        type=int,
        default=64,
        help="Batch size for gibberish detection.",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.90,
        help="Minimum clean-label confidence required to keep text.",
    )

    parser.add_argument(
        "--rewrite-model",
        type=str,
        default="gpt-5.4",
        help="LLM model name for clinical rewriting.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name containing the LLM API key.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--rewrite-batch-size",
        type=int,
        default=15,
        help="Batch size for LLM rewriting.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each LLM API request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM decoding temperature.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum LLM API retries per batch.",
    )
    parser.add_argument(
        "--sleep-between-retries",
        type=float,
        default=2.0,
        help="Sleep time in seconds between LLM retry attempts.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of reusing/resuming them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    """Resolve all input/output paths from CLI arguments."""
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

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
        else data_root / "models" / "question_detector"
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

    output_dir.mkdir(parents=True, exist_ok=True)

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
        step3_clean_texts=output_dir / "step3_clean_texts.jsonl",
        step4_refined=output_dir / "step4_final_refined.jsonl",
        step4_done=output_dir / "step4_refined.done.txt",
    )


def get_torch_device(device_arg: Optional[str]) -> torch.device:
    """Resolve torch device."""
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_file(path: Path, description: str) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    """Raise a clear error if a required directory is missing."""
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file."""
    require_file(path, "JSONL file")
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc

    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_binary_classifier(
    checkpoint_path: Path,
    stats_path: Path,
    device: torch.device,
    in_dim: int = 768,
    width: int = 512,
    num_blocks: int = 4,
) -> Tuple[ResNetBin, np.ndarray, np.ndarray, float]:
    """Load a trained binary classifier and its standardization statistics."""
    require_file(checkpoint_path, "classifier checkpoint")
    require_file(stats_path, "classifier stats")

    model = ResNetBin(in_dim=in_dim, width=width, num_blocks=num_blocks).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    stats = np.load(stats_path)
    if "best_th" in stats.files:
        threshold = float(stats["best_th"][0])
    elif "threshold" in stats.files:
        threshold = float(stats["threshold"][0])
    else:
        raise KeyError(f"No threshold or best_th found in {stats_path}")

    if "mu" not in stats.files or "sg" not in stats.files:
        raise KeyError(f"Expected mu and sg arrays in {stats_path}")

    mu = stats["mu"].astype(np.float32)
    sg = stats["sg"].astype(np.float32)
    sg = np.where(np.abs(sg) < 1e-12, 1.0, sg).astype(np.float32)

    return model, mu, sg, threshold


def find_gmm_clusters(gmm_dir: Path) -> List[str]:
    """Return cluster names inferred from gmm_<cluster>.joblib files."""
    require_dir(gmm_dir, "GMM directory")

    clusters = []
    for path in gmm_dir.glob("gmm_*.joblib"):
        stem = path.stem
        if not stem.startswith("gmm_"):
            continue
        clusters.append(stem.split("_", 1)[1])

    clusters = sorted(set(clusters))
    if not clusters:
        raise FileNotFoundError(f"No gmm_<cluster>.joblib files found in {gmm_dir}")

    return clusters


def load_reference_distribution(reference_csv: Path) -> pd.Series:
    """Load and aggregate the reference ICD-10 chapter distribution into super-clusters."""
    require_file(reference_csv, "reference CSV")

    ref_df = pd.read_csv(reference_csv)
    if "chapter_code" not in ref_df.columns:
        raise ValueError(
            f"Reference CSV must contain a 'chapter_code' column. Found: {list(ref_df.columns)}"
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
    """Allocate integer quotas using the largest-remainder method."""
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
    """Save sampling quota metadata."""
    payload = {
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

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def standardize_embeddings(
    vectors: np.ndarray,
    mu: np.ndarray,
    sg: np.ndarray,
) -> np.ndarray:
    """Apply classifier standardization."""
    return ((vectors - mu) / sg).astype(np.float32)


def predict_classifier_probs(
    model: nn.Module,
    vectors: np.ndarray,
    mu: np.ndarray,
    sg: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict sigmoid probabilities for a batch of embedding vectors."""
    if vectors.size == 0:
        return np.array([], dtype=np.float32)

    standardized = standardize_embeddings(vectors, mu, sg)
    xb = torch.from_numpy(standardized).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(xb)).detach().cpu().numpy()

    return probs.astype(np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors row-wise."""
    denom = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return (vectors / denom).astype(np.float32)


def stratified_sampling(
    paths: PipelinePaths,
    device: torch.device,
    target_total: int,
    candidate_batch_size: int,
    max_trials_min: int,
    max_trials_multiplier: int,
    normalize_samples: bool,
    classifier_in_dim: int,
    classifier_width: int,
    classifier_blocks: int,
    overwrite: bool,
) -> np.ndarray:
    """
    Step 1: Proportional ICD-10 super-cluster sampling from local GMM experts.

    The method:
      - Reads the reference ICD-10 chapter distribution.
      - Maps ICD-10 chapters to coarse super-clusters.
      - Keeps clusters for which a local GMM exists.
      - Allocates target sample counts proportionally.
      - Samples from each local GMM.
      - Keeps only embeddings classified as question-form vectors.
    """
    LOGGER.info("Step 1: ICD-10 proportional local GMM sampling")

    if paths.step1_vectors.exists() and not overwrite:
        LOGGER.info("Loading cached vectors: %s", paths.step1_vectors)
        return np.load(paths.step1_vectors)

    require_dir(paths.gmm_dir, "GMM directory")
    require_file(paths.reference_csv, "reference CSV")

    if overwrite:
        for path in (paths.step1_vectors, paths.step1_meta, paths.step1_quotas):
            if path.exists():
                path.unlink()

    ref_props = load_reference_distribution(paths.reference_csv)
    gmm_clusters = find_gmm_clusters(paths.gmm_dir)

    valid_clusters = [cluster for cluster in gmm_clusters if cluster in ref_props.index]
    if not valid_clusters:
        raise ValueError(
            "No overlapping clusters found between GMM files and reference distribution."
        )

    missing_in_gmm = sorted(set(ref_props.index) - set(valid_clusters))
    if missing_in_gmm:
        LOGGER.warning(
            "Reference clusters without matching GMM will be ignored: %s",
            missing_in_gmm,
        )

    valid_props = ref_props.loc[valid_clusters]
    valid_props = valid_props / valid_props.sum()

    quotas = allocate_largest_remainder(valid_props, target_total)
    save_sampling_quotas(
        path=paths.step1_quotas,
        valid_clusters=valid_clusters,
        valid_props=valid_props,
        quotas=quotas,
        missing_in_gmm=missing_in_gmm,
    )

    LOGGER.info("Sampling quotas:")
    for cluster in valid_clusters:
        LOGGER.info(
            "  %-25s reference=%6.2f%% target_n=%d",
            cluster,
            valid_props[cluster] * 100.0,
            quotas[cluster],
        )

    question_cls, question_mu, question_sg, question_th = load_binary_classifier(
        checkpoint_path=paths.question_model_pt,
        stats_path=paths.question_stats_npz,
        device=device,
        in_dim=classifier_in_dim,
        width=classifier_width,
        num_blocks=classifier_blocks,
    )

    final_vectors: List[np.ndarray] = []
    final_meta: List[Dict[str, Any]] = []

    for cluster_name in valid_clusters:
        target_n = quotas[cluster_name]
        if target_n <= 0:
            continue

        gmm_path = paths.gmm_dir / f"gmm_{cluster_name}.joblib"
        require_file(gmm_path, f"GMM for cluster {cluster_name}")
        gmm = joblib.load(gmm_path)

        accepted = 0
        trials = 0
        max_trials = max(max_trials_min, target_n * max_trials_multiplier)

        pbar = tqdm(total=target_n, desc=f"Cluster {cluster_name}")

        while accepted < target_n and trials < max_trials:
            x_candidate, _ = gmm.sample(candidate_batch_size)
            x_candidate = x_candidate.astype(np.float32)

            if normalize_samples:
                x_candidate = l2_normalize(x_candidate)

            trials += len(x_candidate)

            question_probs = predict_classifier_probs(
                model=question_cls,
                vectors=x_candidate,
                mu=question_mu,
                sg=question_sg,
                device=device,
            )
            x_keep = x_candidate[question_probs >= question_th]

            if x_keep.size == 0:
                continue

            for vector in x_keep:
                if accepted >= target_n:
                    break

                final_vectors.append(vector.astype(np.float32))
                final_meta.append(
                    {
                        "idx": len(final_meta),
                        "source_cluster": cluster_name,
                        "target_quota": int(target_n),
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

    if not final_vectors:
        raise RuntimeError(
            "No vectors were generated. Check GMM files, classifier files, and thresholds."
        )

    x_final = np.vstack(final_vectors).astype(np.float32)
    np.save(paths.step1_vectors, x_final)
    append_jsonl(paths.step1_meta, final_meta)

    LOGGER.info("Saved vectors: %s", paths.step1_vectors)
    LOGGER.info("Saved metadata: %s", paths.step1_meta)
    LOGGER.info("Final vector shape: %s", x_final.shape)

    return x_final


def load_done_indices_from_jsonl(path: Path) -> set[int]:
    """Load existing idx values from a JSONL output file."""
    if not path.exists():
        return set()

    done: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done.add(int(row["idx"]))
            except Exception:
                continue

    return done


def invert_vectors(
    paths: PipelinePaths,
    vectors: np.ndarray,
    device: torch.device,
    corrector_name: str,
    batch_size: int,
    num_steps: int,
    overwrite: bool,
) -> None:
    """Step 2: Decode embedding vectors into raw text using vec2text."""
    LOGGER.info("Step 2: vec2text inversion")

    if paths.step2_texts.exists() and overwrite:
        paths.step2_texts.unlink()

    done_indices = load_done_indices_from_jsonl(paths.step2_texts)
    if len(done_indices) >= len(vectors):
        LOGGER.info("Step 2 already complete: %s", paths.step2_texts)
        return

    try:
        import vec2text  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "vec2text is required for Step 2. Install it in the review environment "
            "or run with --steps excluding 'invert'."
        ) from exc

    corrector = vec2text.load_pretrained_corrector(corrector_name)

    for start in tqdm(range(0, len(vectors), batch_size), desc="Inverting"):
        end = min(start + batch_size, len(vectors))
        batch_indices = list(range(start, end))
        pending = [idx for idx in batch_indices if idx not in done_indices]

        if not pending:
            continue

        emb_batch = torch.from_numpy(vectors[pending]).to(device)

        with torch.no_grad():
            decoded = vec2text.invert_embeddings(
                embeddings=emb_batch,
                corrector=corrector,
                num_steps=num_steps,
            )

        rows = [
            {"idx": int(idx), "text": str(text)}
            for idx, text in zip(pending, decoded)
        ]
        append_jsonl(paths.step2_texts, rows)

    LOGGER.info("Saved raw decoded texts: %s", paths.step2_texts)


def copy_step2_to_step3(paths: PipelinePaths, overwrite: bool) -> None:
    """Copy Step 2 output to Step 3 output when gibberish filtering is disabled."""
    require_file(paths.step2_texts, "Step 2 text output")

    if paths.step3_clean_texts.exists() and not overwrite:
        LOGGER.info("Step 3 output already exists: %s", paths.step3_clean_texts)
        return

    shutil.copyfile(paths.step2_texts, paths.step3_clean_texts)
    LOGGER.info("Copied %s to %s", paths.step2_texts, paths.step3_clean_texts)


def gibberish_filtering(
    paths: PipelinePaths,
    model_name: str,
    batch_size: int,
    threshold: float,
    overwrite: bool,
) -> None:
    """Step 3: Filter raw decoded text using a text-based gibberish detector."""
    LOGGER.info("Step 3: gibberish filtering")

    require_file(paths.step2_texts, "Step 2 text output")

    if paths.step3_clean_texts.exists() and not overwrite:
        LOGGER.info("Step 3 output already exists: %s", paths.step3_clean_texts)
        return

    if paths.step3_clean_texts.exists() and overwrite:
        paths.step3_clean_texts.unlink()

    rows = read_jsonl(paths.step2_texts)
    if not rows:
        raise ValueError(f"No rows found in {paths.step2_texts}")

    texts = [str(row["text"]) for row in rows]
    device_id = 0 if torch.cuda.is_available() else -1

    detector = pipeline(
        "text-classification",
        model=model_name,
        device=device_id,
    )

    # Preserve the original heuristic: infer the clean label by testing a simple
    # valid medical question.
    clean_probe = detector("What is the treatment?")[0]
    clean_label = clean_probe["label"]

    kept = 0
    for start in tqdm(range(0, len(texts), batch_size), desc="Gibberish check"):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        results = detector(batch, truncation=True)

        out_rows: List[Dict[str, Any]] = []
        for offset, result in enumerate(results):
            source_row = rows[start + offset]
            is_clean = result["label"] == clean_label and float(result["score"]) >= threshold
            if is_clean:
                out_rows.append(
                    {
                        "idx": int(source_row["idx"]),
                        "text": batch[offset],
                        "gib_score": float(result["score"]),
                    }
                )

        if out_rows:
            append_jsonl(paths.step3_clean_texts, out_rows)
            kept += len(out_rows)

    LOGGER.info("Saved clean decoded texts: %s", paths.step3_clean_texts)
    LOGGER.info("Kept %d/%d texts after gibberish filtering.", kept, len(texts))


def load_done_ids(path: Path) -> set[str]:
    """Load completed IDs from a plain-text done file."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_done_ids(path: Path, ids: Iterable[Any]) -> None:
    """Append completed IDs to a plain-text done file."""
    with path.open("a", encoding="utf-8") as f:
        for item in ids:
            f.write(str(item) + "\n")


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from an LLM response.

    The prompt requests strict JSON, but this fallback handles occasional extra text.
    """
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object.")

    return obj


def validate_rewrite_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize required result fields without changing the rewriting method."""
    if "idx" not in result:
        raise ValueError(f"Missing idx in result: {result}")

    normalized = {
        "idx": int(result["idx"]),
        "rewritten": result.get("rewritten"),
        "is_medical": bool(result.get("is_medical", False)),
        "correction_made": bool(result.get("correction_made", False)),
        "reason": str(result.get("reason", "")),
    }

    if normalized["rewritten"] is not None:
        rewritten = str(normalized["rewritten"]).strip()
        normalized["rewritten"] = rewritten if rewritten else None

    return normalized


def make_openai_client(
    api_key_env: str,
    base_url: Optional[str],
    timeout: float,
) -> Any:
    """Create an OpenAI-compatible client using an API key from the environment."""
    if OpenAI is None:
        raise ImportError(
            "The openai package is required for Step 4. Install it or skip the refine step."
        )

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set environment variable {api_key_env} before running refinement."
        )

    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def call_rewrite_model(
    client: Any,
    model: str,
    batch: Sequence[Dict[str, Any]],
    temperature: float,
) -> List[Dict[str, Any]]:
    """Call the LLM rewriting model for one batch."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"items": list(batch)}, ensure_ascii=False)},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response content is empty.")

    payload = extract_json_object(content)
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"LLM response does not contain a results list: {payload}")

    return [validate_rewrite_result(row) for row in payload["results"]]


def refine_texts(
    paths: PipelinePaths,
    api_key_env: str,
    base_url: Optional[str],
    model: str,
    batch_size: int,
    request_timeout: float,
    temperature: float,
    max_retries: int,
    sleep_between_retries: float,
    overwrite: bool,
) -> None:
    """Step 4: LLM-based clinical rewriting and validation."""
    LOGGER.info("Step 4: clinical rewriting / validation")

    require_file(paths.step3_clean_texts, "Step 3 clean text output")

    if overwrite:
        for path in (paths.step4_refined, paths.step4_done):
            if path.exists():
                path.unlink()

    done_ids = load_done_ids(paths.step4_done)
    data = read_jsonl(paths.step3_clean_texts)
    pending_data = [row for row in data if str(row["idx"]) not in done_ids]

    if not pending_data:
        LOGGER.info("Step 4 already complete: %s", paths.step4_refined)
        return

    client = make_openai_client(
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=request_timeout,
    )

    for start in tqdm(range(0, len(pending_data), batch_size), desc="Refining"):
        batch = pending_data[start : start + batch_size]

        last_error: Optional[Exception] = None
        results: Optional[List[Dict[str, Any]]] = None

        for attempt in range(1, max_retries + 1):
            try:
                results = call_rewrite_model(
                    client=client,
                    model=model,
                    batch=batch,
                    temperature=temperature,
                )
                break
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Rewrite batch failed on attempt %d/%d: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(sleep_between_retries)

        if results is None:
            LOGGER.error("Skipping batch after repeated failures. Last error: %s", last_error)
            continue

        append_jsonl(paths.step4_refined, results)
        append_done_ids(paths.step4_done, [row["idx"] for row in results])

    LOGGER.info("Saved refined output: %s", paths.step4_refined)


def load_vectors_for_later_step(paths: PipelinePaths) -> np.ndarray:
    """Load Step 1 vectors when running later steps directly."""
    require_file(paths.step1_vectors, "Step 1 vector output")
    return np.load(paths.step1_vectors)


def normalize_steps(raw_steps: Sequence[str]) -> List[str]:
    """Expand 'all' into explicit pipeline steps."""
    if "all" in raw_steps:
        return ["sample", "invert", "gibberish", "refine"]

    ordered = ["sample", "invert", "gibberish", "refine"]
    requested = set(raw_steps)
    return [step for step in ordered if step in requested]


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    seed_everything(args.seed)

    paths = resolve_paths(args)
    device = get_torch_device(args.device)
    steps = normalize_steps(args.steps)

    LOGGER.info("Output directory: %s", paths.output_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Steps: %s", steps)

    vectors: Optional[np.ndarray] = None

    if "sample" in steps:
        vectors = stratified_sampling(
            paths=paths,
            device=device,
            target_total=args.target_total,
            candidate_batch_size=args.candidate_batch_size,
            max_trials_min=args.max_trials_min,
            max_trials_multiplier=args.max_trials_multiplier,
            normalize_samples=not args.no_normalize_samples,
            classifier_in_dim=args.embedding_dim,
            classifier_width=args.classifier_width,
            classifier_blocks=args.classifier_blocks,
            overwrite=args.overwrite,
        )

    if "invert" in steps:
        if vectors is None:
            vectors = load_vectors_for_later_step(paths)

        invert_vectors(
            paths=paths,
            vectors=vectors,
            device=device,
            corrector_name=args.vec2text_corrector,
            batch_size=args.vec2text_batch_size,
            num_steps=args.vec2text_num_steps,
            overwrite=args.overwrite,
        )

    if "gibberish" in steps:
        if args.disable_gibberish_filter:
            LOGGER.info("Gibberish filtering disabled by CLI flag.")
            copy_step2_to_step3(paths=paths, overwrite=args.overwrite)
        else:
            gibberish_filtering(
                paths=paths,
                model_name=args.gibberish_model,
                batch_size=args.gibberish_batch_size,
                threshold=args.gibberish_threshold,
                overwrite=args.overwrite,
            )

    if "refine" in steps:
        refine_texts(
            paths=paths,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.rewrite_model,
            batch_size=args.rewrite_batch_size,
            request_timeout=args.request_timeout,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sleep_between_retries=args.sleep_between_retries,
            overwrite=args.overwrite,
        )

    LOGGER.info("Pipeline completed. Results saved in %s", paths.output_dir)


if __name__ == "__main__":
    main()