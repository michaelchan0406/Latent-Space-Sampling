#!/usr/bin/env python3
"""
Reviewer-friendly ablation pipeline: Global GMM instead of ICD-10 local experts.

This script implements the "No ICD Local Experts / Global GMM" ablation.

Compared with the full pipeline, the intended experimental change is:
  - Full pipeline: ICD-10-based local GMM experts.
  - This ablation: one global GMM trained on L2-normalized MedQuAD + PubMedQA
    embeddings.

Pipeline:
  1. Fit or load one global GMM on L2-normalized reference embeddings.
  2. Sample latent vectors from the global GMM.
  3. L2-normalize sampled vectors.
  4. Filter sampled vectors using the L2 question-form classifier.
  5. Decode accepted vectors into text using vec2text.
  6. Re-embed decoded text with GTR-T5-base and L2-normalize.
  7. Filter decoded text using a gibberish detector.
  8. Rewrite / validate remaining text using an LLM.

Removed from this ablation:
  - ICD-10 local experts.
  - Local GMMs.
  - FATE / medical-only classifier.
  - Pre-rewrite LLM medical-only filtering.

No private paths, names, institutions, or API keys are embedded in this file.
Supply API credentials through environment variables.

Example:
  python ablation_global_gmm.py \\
    --data-root ./data \\
    --output-dir ./outputs/ablation_global_gmm \\
    --steps all

Example, force re-training the global GMM:
  python ablation_global_gmm.py \\
    --data-root ./data \\
    --output-dir ./outputs/ablation_global_gmm \\
    --steps sample \\
    --force-refit-gmm

Expected default input layout:
  data/
    embeddings_gtr_t5_base/
      medquad_questions.gtr_t5_base.float32.npy
      pubmed_qa_unlabeled.gtr_t5_base.float32.npy
    models/
      question_detector_gtr_t5_base_resnet_L2/
        resnet_qvstmt_gtr_t5_base_L2.pt
        standardize_mu_sg_and_threshold_L2.npz

Outputs:
  output-dir/
    run_config.json
    global_medquad_pubmedqa_gmm_60comp_L2.joblib
    step1_global_gmm_L2_question_filtered_vectors.npy
    step1_global_gmm_L2_question_filtered_meta.jsonl
    step2_vec2text_texts.jsonl
    step3_reembedded_vectors_L2.npy
    step3_reembedded_meta.jsonl
    step4_gibberish_kept_vectors_L2.npy
    step4_gibberish_kept_texts.jsonl
    step5_llm_rewritten_questions.jsonl
    step5_llm_rewrite.done.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


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

    medquad_embeddings: Path
    pubmedqa_embeddings: Path

    question_model_pt: Path
    question_stats_npz: Path

    global_gmm: Path

    step1_vectors: Path
    step1_meta: Path

    step2_texts: Path

    step3_reembedded_vectors: Path
    step3_reembedded_meta: Path

    step4_kept_vectors: Path
    step4_kept_texts: Path

    step5_rewritten: Path
    step5_done: Path

    run_config: Path


@dataclass(frozen=True)
class GlobalGMMConfig:
    seed: int
    embedding_dim: int

    target_raw: int
    batch_sample: int
    max_trials: int

    gmm_components: int
    gmm_covariance_type: str
    gmm_reg_covar: float
    gmm_max_iter: int
    gmm_n_init: int

    vec2text_corrector: str
    vec2text_batch_size: int
    vec2text_steps: int
    vec2text_beam: int

    reembed_model: str
    reembed_batch_size: int
    reembed_max_length: int

    gibberish_model: str
    gibberish_batch_size: int
    gibberish_threshold: float

    rewrite_model: str
    rewrite_batch_size: int
    rewrite_temperature: float
    rewrite_max_retries: int
    rewrite_timeout: float

    normalization: str = (
        "L2 before global GMM training, after GMM sampling, before question "
        "classification, before vec2text, and after re-embedding."
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
        proj_dropout: float = 0.30,
        block_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, p=block_dropout) for _ in range(num_blocks)]
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
        description="Global GMM ablation pipeline for synthetic medical question generation."
    )

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--medquad-embeddings", type=Path, default=None)
    parser.add_argument("--pubmedqa-embeddings", type=Path, default=None)
    parser.add_argument("--question-model-dir", type=Path, default=None)
    parser.add_argument("--question-model-pt", type=Path, default=None)
    parser.add_argument("--question-stats-npz", type=Path, default=None)
    parser.add_argument("--global-gmm-path", type=Path, default=None)

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "sample", "invert", "reembed", "gibberish", "rewrite"],
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--embedding-dim", type=int, default=768)

    parser.add_argument("--target-raw", type=int, default=10000)
    parser.add_argument("--batch-sample", type=int, default=1000)
    parser.add_argument("--max-trials", type=int, default=500000)

    parser.add_argument("--gmm-components", type=int, default=60)
    parser.add_argument("--gmm-covariance-type", type=str, default="full")
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-5)
    parser.add_argument("--gmm-max-iter", type=int, default=300)
    parser.add_argument("--gmm-n-init", type=int, default=1)
    parser.add_argument(
        "--force-refit-gmm",
        action="store_true",
        help="Train the global GMM even if a cached GMM exists.",
    )

    parser.add_argument("--vec2text-corrector", type=str, default="gtr-base")
    parser.add_argument("--vec2text-batch-size", type=int, default=40)
    parser.add_argument("--vec2text-steps", type=int, default=40)
    parser.add_argument("--vec2text-beam", type=int, default=4)

    parser.add_argument(
        "--reembed-model",
        type=str,
        default="sentence-transformers/gtr-t5-base",
    )
    parser.add_argument("--reembed-batch-size", type=int, default=256)
    parser.add_argument("--reembed-max-length", type=int, default=128)

    parser.add_argument(
        "--gibberish-model",
        type=str,
        default="madhurjindal/autonlp-Gibberish-Detector-492513457",
    )
    parser.add_argument("--gibberish-batch-size", type=int, default=64)
    parser.add_argument("--gibberish-threshold", type=float, default=0.90)

    parser.add_argument("--rewrite-model", type=str, default="gpt-5.4")
    parser.add_argument("--rewrite-batch-size", type=int, default=20)
    parser.add_argument("--rewrite-temperature", type=float, default=0.2)
    parser.add_argument("--rewrite-max-retries", type=int, default=5)
    parser.add_argument("--rewrite-timeout", type=float, default=60.0)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", type=str, default="OPENAI_BASE_URL")
    parser.add_argument("--base-url", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def normalize_steps(raw_steps: Sequence[str]) -> List[str]:
    if "all" in raw_steps:
        return ["sample", "invert", "reembed", "gibberish", "rewrite"]

    ordered = ["sample", "invert", "reembed", "gibberish", "rewrite"]
    requested = set(raw_steps)
    return [step for step in ordered if step in requested]


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = data_root / "embeddings_gtr_t5_base"

    medquad_embeddings = (
        args.medquad_embeddings.expanduser().resolve()
        if args.medquad_embeddings is not None
        else embeddings_dir / "medquad_questions.gtr_t5_base.float32.npy"
    )
    pubmedqa_embeddings = (
        args.pubmedqa_embeddings.expanduser().resolve()
        if args.pubmedqa_embeddings is not None
        else embeddings_dir / "pubmed_qa_unlabeled.gtr_t5_base.float32.npy"
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

    global_gmm = (
        args.global_gmm_path.expanduser().resolve()
        if args.global_gmm_path is not None
        else output_dir / f"global_medquad_pubmedqa_gmm_{args.gmm_components}comp_L2.joblib"
    )

    return PipelinePaths(
        data_root=data_root,
        output_dir=output_dir,
        medquad_embeddings=medquad_embeddings,
        pubmedqa_embeddings=pubmedqa_embeddings,
        question_model_pt=question_model_pt,
        question_stats_npz=question_stats_npz,
        global_gmm=global_gmm,
        step1_vectors=output_dir / "step1_global_gmm_L2_question_filtered_vectors.npy",
        step1_meta=output_dir / "step1_global_gmm_L2_question_filtered_meta.jsonl",
        step2_texts=output_dir / "step2_vec2text_texts.jsonl",
        step3_reembedded_vectors=output_dir / "step3_reembedded_vectors_L2.npy",
        step3_reembedded_meta=output_dir / "step3_reembedded_meta.jsonl",
        step4_kept_vectors=output_dir / "step4_gibberish_kept_vectors_L2.npy",
        step4_kept_texts=output_dir / "step4_gibberish_kept_texts.jsonl",
        step5_rewritten=output_dir / "step5_llm_rewritten_questions.jsonl",
        step5_done=output_dir / "step5_llm_rewrite.done.txt",
        run_config=output_dir / "run_config.json",
    )


def make_config(args: argparse.Namespace) -> GlobalGMMConfig:
    return GlobalGMMConfig(
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        target_raw=args.target_raw,
        batch_sample=args.batch_sample,
        max_trials=args.max_trials,
        gmm_components=args.gmm_components,
        gmm_covariance_type=args.gmm_covariance_type,
        gmm_reg_covar=args.gmm_reg_covar,
        gmm_max_iter=args.gmm_max_iter,
        gmm_n_init=args.gmm_n_init,
        vec2text_corrector=args.vec2text_corrector,
        vec2text_batch_size=args.vec2text_batch_size,
        vec2text_steps=args.vec2text_steps,
        vec2text_beam=args.vec2text_beam,
        reembed_model=args.reembed_model,
        reembed_batch_size=args.reembed_batch_size,
        reembed_max_length=args.reembed_max_length,
        gibberish_model=args.gibberish_model,
        gibberish_batch_size=args.gibberish_batch_size,
        gibberish_threshold=args.gibberish_threshold,
        rewrite_model=args.rewrite_model,
        rewrite_batch_size=args.rewrite_batch_size,
        rewrite_temperature=args.rewrite_temperature,
        rewrite_max_retries=args.rewrite_max_retries,
        rewrite_timeout=args.rewrite_timeout,
    )


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_file(path: Path, description: str) -> None:
    if not path.exists():
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


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def load_question_classifier(
    model_pt: Path,
    stats_npz: Path,
    embedding_dim: int,
    device: torch.device,
) -> Tuple[ResNetBin, np.ndarray, np.ndarray, float]:
    require_file(model_pt, "question classifier checkpoint")
    require_file(stats_npz, "question classifier stats")

    model = ResNetBin(in_dim=embedding_dim)
    model.load_state_dict(torch.load(model_pt, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    stats = np.load(stats_npz)
    if "mu" not in stats.files or "sg" not in stats.files:
        raise KeyError(f"Expected 'mu' and 'sg' arrays in {stats_npz}")

    mu = stats["mu"].squeeze().astype(np.float32)
    sg = stats["sg"].squeeze().astype(np.float32)

    if "best_th" in stats.files:
        threshold = float(stats["best_th"][0])
    elif "threshold" in stats.files:
        threshold = float(stats["threshold"][0])
    else:
        raise KeyError(f"Expected 'best_th' or 'threshold' in {stats_npz}")

    if mu.shape[0] != embedding_dim or sg.shape[0] != embedding_dim:
        raise ValueError(
            f"Classifier stats dimension mismatch: mu={mu.shape}, sg={sg.shape}, "
            f"expected={embedding_dim}"
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
    """
    Predict question-form probabilities.

    The input x should already be L2-normalized, matching the classifier's
    training space.
    """
    model.eval()
    probs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = (x[start : start + batch_size].astype(np.float32) - mu) / sg
            xb = torch.tensor(batch, dtype=torch.float32, device=device)
            logits = model(xb).detach().cpu().numpy()
            probs.append(sigmoid_np(logits))

    return np.concatenate(probs, axis=0)


def load_reference_embeddings(paths: PipelinePaths, embedding_dim: int) -> np.ndarray:
    """Load and L2-normalize embeddings used to train the global GMM."""
    require_file(paths.medquad_embeddings, "MedQuAD embedding file")

    LOGGER.info("Loading MedQuAD embeddings: %s", paths.medquad_embeddings)
    medquad = np.load(paths.medquad_embeddings).astype(np.float32)
    log_norm_stats("MedQuAD raw", medquad)
    medquad = l2_normalize(medquad)
    log_norm_stats("MedQuAD L2", medquad)

    arrays = [medquad]

    if paths.pubmedqa_embeddings.exists():
        LOGGER.info("Loading PubMedQA embeddings: %s", paths.pubmedqa_embeddings)
        pubmedqa = np.load(paths.pubmedqa_embeddings).astype(np.float32)
        log_norm_stats("PubMedQA raw", pubmedqa)
        pubmedqa = l2_normalize(pubmedqa)
        log_norm_stats("PubMedQA L2", pubmedqa)
        arrays.append(pubmedqa)
    else:
        LOGGER.warning(
            "PubMedQA embeddings not found. Training global GMM on MedQuAD only: %s",
            paths.pubmedqa_embeddings,
        )

    combined = np.vstack(arrays).astype(np.float32)
    combined = l2_normalize(combined)

    if combined.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {combined.shape[1]}, expected {embedding_dim}"
        )

    LOGGER.info("Combined reference embeddings: shape=%s", combined.shape)
    log_norm_stats("Combined L2 for global GMM training", combined)

    return combined


def fit_or_load_global_gmm(
    paths: PipelinePaths,
    config: GlobalGMMConfig,
    force_refit: bool,
) -> GaussianMixture:
    """
    Fit or load the global L2-space GMM.

    This is the key ablation component:
      ICD-10 local GMM experts are replaced by a single global GMM.
    """
    if paths.global_gmm.exists() and not force_refit:
        LOGGER.info("Loading existing global GMM: %s", paths.global_gmm)
        return joblib.load(paths.global_gmm)

    embeddings = load_reference_embeddings(paths, config.embedding_dim)

    if embeddings.shape[0] < config.gmm_components:
        raise ValueError(
            f"Not enough embeddings ({embeddings.shape[0]}) for "
            f"{config.gmm_components} GMM components."
        )

    LOGGER.info(
        "Fitting global L2 GMM: n_components=%d covariance_type=%s",
        config.gmm_components,
        config.gmm_covariance_type,
    )

    gmm = GaussianMixture(
        n_components=config.gmm_components,
        covariance_type=config.gmm_covariance_type,
        random_state=config.seed,
        reg_covar=config.gmm_reg_covar,
        max_iter=config.gmm_max_iter,
        n_init=config.gmm_n_init,
        verbose=1,
    )
    gmm.fit(embeddings)

    LOGGER.info("Global GMM converged: %s", bool(gmm.converged_))
    LOGGER.info("Global GMM lower bound: %.6f", float(gmm.lower_bound_))

    joblib.dump(gmm, paths.global_gmm)
    LOGGER.info("Saved global GMM: %s", paths.global_gmm)

    del embeddings
    gc.collect()

    return gmm


def step1_global_gmm_sampling(
    paths: PipelinePaths,
    config: GlobalGMMConfig,
    device: torch.device,
    force_refit_gmm: bool,
    overwrite: bool,
) -> np.ndarray:
    """
    Step 1: Global GMM sampling + question-form filtering.

    This differs from the full pipeline by replacing ICD-10 local experts with
    a single global GMM trained on pooled reference embeddings.
    """
    LOGGER.info("Step 1: Global L2 GMM sampling + question-form filtering")

    if (
        paths.step1_vectors.exists()
        and paths.step1_meta.exists()
        and not overwrite
    ):
        x_cached = l2_normalize(np.load(paths.step1_vectors).astype(np.float32))
        LOGGER.info("Loaded cached Step 1 vectors: %s shape=%s", paths.step1_vectors, x_cached.shape)
        log_norm_stats("Cached Step 1 vectors", x_cached)
        return x_cached

    if overwrite:
        for path in (paths.step1_vectors, paths.step1_meta):
            if path.exists():
                path.unlink()

    gmm = fit_or_load_global_gmm(paths, config, force_refit=force_refit_gmm)

    q_model, q_mu, q_sg, q_th = load_question_classifier(
        model_pt=paths.question_model_pt,
        stats_npz=paths.question_stats_npz,
        embedding_dim=config.embedding_dim,
        device=device,
    )
    LOGGER.info("Question classifier threshold: %.6f", q_th)

    kept_batches: List[np.ndarray] = []
    meta_records: List[Dict[str, Any]] = []

    accepted_total = 0
    trials = 0

    pbar = tqdm(
        total=config.target_raw,
        desc="Global GMM sampling + question filtering",
    )

    while accepted_total < config.target_raw and trials < config.max_trials:
        x_candidate, component_ids = gmm.sample(config.batch_sample)
        x_candidate = x_candidate.astype(np.float32)

        # Gaussian samples are not guaranteed to remain on the unit sphere.
        # Project them back to the L2-normalized embedding space before filtering.
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
        component_keep = component_ids[keep_mask]
        q_keep = q_probs[keep_mask]

        if x_keep.shape[0] == 0:
            continue

        remaining = config.target_raw - accepted_total
        x_keep = x_keep[:remaining]
        component_keep = component_keep[:remaining]
        q_keep = q_keep[:remaining]

        kept_batches.append(x_keep.astype(np.float32))
        accepted_total += x_keep.shape[0]
        pbar.update(x_keep.shape[0])

        for comp_id, q_score in zip(component_keep, q_keep):
            meta_records.append(
                {
                    "idx": len(meta_records),
                    "source": "global_gmm_L2",
                    "gmm_component": int(comp_id),
                    "q_prob": float(q_score),
                }
            )

    pbar.close()

    if accepted_total < config.target_raw:
        LOGGER.warning(
            "Only accepted %d/%d vectors after %d sampled candidates.",
            accepted_total,
            config.target_raw,
            trials,
        )

    if not kept_batches:
        raise RuntimeError(
            "No vectors passed the question classifier. Check the GMM, threshold, and embeddings."
        )

    x_raw = np.vstack(kept_batches).astype(np.float32)
    x_raw = l2_normalize(x_raw)
    meta_records = meta_records[: x_raw.shape[0]]

    np.save(paths.step1_vectors, x_raw)
    append_jsonl(paths.step1_meta, meta_records)

    LOGGER.info("Saved Step 1 vectors: %s shape=%s", paths.step1_vectors, x_raw.shape)
    LOGGER.info("Saved Step 1 metadata: %s", paths.step1_meta)
    LOGGER.info("Step 1 acceptance rate: %.6f", x_raw.shape[0] / max(trials, 1))
    log_norm_stats("Step 1 saved vectors", x_raw)

    return x_raw


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
    config: GlobalGMMConfig,
    device: torch.device,
    overwrite: bool,
) -> None:
    """Step 2: Decode accepted vectors into raw text using vec2text."""
    LOGGER.info("Step 2: vec2text inversion")

    require_file(paths.step1_vectors, "Step 1 vector file")

    if overwrite and paths.step2_texts.exists():
        paths.step2_texts.unlink()

    x_target = l2_normalize(np.load(paths.step1_vectors).astype(np.float32))
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


def mean_pool_encoder_output(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled


def step3_reembed_decoded_texts(
    paths: PipelinePaths,
    config: GlobalGMMConfig,
    device: torch.device,
    overwrite: bool,
) -> np.ndarray:
    """Step 3: Re-embed decoded texts with GTR-T5-base and L2-normalize."""
    LOGGER.info("Step 3: Re-embed decoded texts")

    if (
        paths.step3_reembedded_vectors.exists()
        and paths.step3_reembedded_meta.exists()
        and not overwrite
    ):
        e_cached = l2_normalize(np.load(paths.step3_reembedded_vectors).astype(np.float32))
        LOGGER.info(
            "Loaded cached re-embedded vectors: %s shape=%s",
            paths.step3_reembedded_vectors,
            e_cached.shape,
        )
        log_norm_stats("Cached re-embedded vectors", e_cached)
        return e_cached

    if overwrite:
        for path in (paths.step3_reembedded_vectors, paths.step3_reembedded_meta):
            if path.exists():
                path.unlink()

    rows = read_jsonl(paths.step2_texts)
    rows = sorted(rows, key=lambda r: int(r["idx"]))

    texts = [str(row.get("text", "")) for row in rows]
    idxs = [int(row["idx"]) for row in rows]

    LOGGER.info("Re-embedding %d decoded texts with %s", len(texts), config.reembed_model)

    tokenizer = AutoTokenizer.from_pretrained(config.reembed_model)
    encoder = AutoModel.from_pretrained(config.reembed_model).encoder.to(device).eval()

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), config.reembed_batch_size),
            desc="Re-embedding",
        ):
            batch = texts[start : start + config.reembed_batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.reembed_max_length,
            ).to(device)

            output = encoder(**inputs)
            pooled = mean_pool_encoder_output(
                last_hidden_state=output.last_hidden_state,
                attention_mask=inputs["attention_mask"],
            )
            embeddings = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))

            if device.type == "cuda":
                torch.cuda.empty_cache()

    e_reembedded = np.vstack(all_embeddings).astype(np.float32)
    e_reembedded = l2_normalize(e_reembedded)

    np.save(paths.step3_reembedded_vectors, e_reembedded)

    meta_rows = [
        {
            "row_id": row_id,
            "idx": idx,
        }
        for row_id, idx in enumerate(idxs)
    ]
    append_jsonl(paths.step3_reembedded_meta, meta_rows)

    LOGGER.info(
        "Saved Step 3 re-embedded vectors: %s shape=%s",
        paths.step3_reembedded_vectors,
        e_reembedded.shape,
    )
    LOGGER.info("Saved Step 3 metadata: %s", paths.step3_reembedded_meta)
    log_norm_stats("Step 3 saved re-embedded vectors", e_reembedded)

    return e_reembedded


def step4_gibberish_detection(
    paths: PipelinePaths,
    config: GlobalGMMConfig,
    device: torch.device,
    overwrite: bool,
) -> np.ndarray:
    """Step 4: Filter decoded texts using a gibberish detector."""
    LOGGER.info("Step 4: Gibberish detection")

    if (
        paths.step4_kept_vectors.exists()
        and paths.step4_kept_texts.exists()
        and not overwrite
    ):
        e_cached = l2_normalize(np.load(paths.step4_kept_vectors).astype(np.float32))
        LOGGER.info(
            "Loaded cached gibberish-kept vectors: %s shape=%s",
            paths.step4_kept_vectors,
            e_cached.shape,
        )
        LOGGER.info(
            "Loaded cached gibberish-kept texts: %s n=%d",
            paths.step4_kept_texts,
            count_jsonl(paths.step4_kept_texts),
        )
        log_norm_stats("Cached gibberish-kept vectors", e_cached)
        return e_cached

    if overwrite:
        for path in (paths.step4_kept_vectors, paths.step4_kept_texts):
            if path.exists():
                path.unlink()

    rows = read_jsonl(paths.step2_texts)
    rows = sorted(rows, key=lambda r: int(r["idx"]))
    texts = [str(row.get("text", "")) for row in rows]

    require_file(paths.step3_reembedded_vectors, "Step 3 re-embedded vectors")
    e_reembedded = l2_normalize(np.load(paths.step3_reembedded_vectors).astype(np.float32))

    if e_reembedded.shape[0] != len(rows):
        raise ValueError(
            f"Length mismatch: re-embedded vectors={e_reembedded.shape[0]}, "
            f"decoded text rows={len(rows)}"
        )

    detector = pipeline(
        "text-classification",
        model=config.gibberish_model,
        device=0 if device.type == "cuda" else -1,
    )

    clean_probe = "What is the recommended treatment for hypertension?"
    clean_label = detector(clean_probe)[0]["label"]
    LOGGER.info("Inferred clean gibberish-detector label: %s", clean_label)

    gib_labels: List[str] = []
    gib_scores: List[float] = []

    for start in tqdm(
        range(0, len(texts), config.gibberish_batch_size),
        desc="Gibberish check",
    ):
        batch = texts[start : start + config.gibberish_batch_size]
        results = detector(batch, truncation=True)

        gib_labels.extend([str(r["label"]) for r in results])
        gib_scores.extend([float(r["score"]) for r in results])

    keep_mask = np.array(
        [
            label == clean_label and score >= config.gibberish_threshold
            for label, score in zip(gib_labels, gib_scores)
        ],
        dtype=bool,
    )

    e_keep = l2_normalize(e_reembedded[keep_mask])

    np.save(paths.step4_kept_vectors, e_keep)

    kept_text_rows: List[Dict[str, Any]] = []
    for row_id, keep in enumerate(keep_mask):
        if not keep:
            continue

        kept_text_rows.append(
            {
                "idx": int(rows[row_id]["idx"]),
                "text": texts[row_id],
                "gib_label": gib_labels[row_id],
                "gib_score": float(gib_scores[row_id]),
            }
        )

    append_jsonl(paths.step4_kept_texts, kept_text_rows)

    LOGGER.info("Step 4 kept %d/%d decoded texts.", int(keep_mask.sum()), len(texts))
    LOGGER.info("Saved Step 4 kept vectors: %s", paths.step4_kept_vectors)
    LOGGER.info("Saved Step 4 kept texts: %s", paths.step4_kept_texts)
    log_norm_stats("Step 4 saved gibberish-kept vectors", e_keep)

    return e_keep


def make_openai_client(
    api_key_env: str,
    base_url_env: str,
    base_url_arg: Optional[str],
    timeout: float,
) -> Any:
    if OpenAI is None:
        raise ImportError(
            "The openai package is required for the rewrite step. Install it or skip 'rewrite'."
        )

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set environment variable {api_key_env} before running rewrite."
        )

    base_url = base_url_arg or os.environ.get(base_url_env)

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


def load_done_idx(done_path: Path, output_path: Path) -> Set[int]:
    done: Set[int] = set()

    if done_path.exists():
        with done_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    try:
                        done.add(int(s))
                    except Exception:
                        continue

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "idx" in obj:
                        done.add(int(obj["idx"]))
                except Exception:
                    continue

    return done


def normalize_rewrite_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "idx" not in raw:
        raise ValueError(f"Missing idx in rewrite result: {raw}")

    rewritten = raw.get("rewritten", None)
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

            normalized = [normalize_rewrite_result(r) for r in results if isinstance(r, dict)]

            got_idx = {int(r["idx"]) for r in normalized}
            expected_idx = set(batch_idx)

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

            sleep_s = min(2 ** attempt, 20)
            LOGGER.warning(
                "Rewrite batch failed on attempt %d/%d: %s. Sleeping %ds.",
                attempt,
                max_retries,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError("Unreachable rewrite retry state.")


def step5_llm_rewrite(
    paths: PipelinePaths,
    config: GlobalGMMConfig,
    api_key_env: str,
    base_url_env: str,
    base_url_arg: Optional[str],
    overwrite: bool,
) -> None:
    """Step 5: LLM-based clinical rewrite / validation."""
    LOGGER.info("Step 5: LLM-based clinical rewrite")

    require_file(paths.step4_kept_texts, "Step 4 kept text file")

    if overwrite:
        for path in (paths.step5_rewritten, paths.step5_done):
            if path.exists():
                path.unlink()

    client = make_openai_client(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        base_url_arg=base_url_arg,
        timeout=config.rewrite_timeout,
    )

    done_idx = load_done_idx(paths.step5_done, paths.step5_rewritten)

    total = count_jsonl(paths.step4_kept_texts)
    LOGGER.info("Rewrite input: %s", paths.step4_kept_texts)
    LOGGER.info("Rewrite output: %s", paths.step5_rewritten)
    LOGGER.info("Already done: %d", len(done_idx))
    LOGGER.info("Rewrite model: %s", config.rewrite_model)

    batch_in: List[Dict[str, Any]] = []
    batch_raw: List[Dict[str, Any]] = []

    output_mode = "a" if paths.step5_rewritten.exists() else "w"

    with paths.step4_kept_texts.open("r", encoding="utf-8") as fin, \
        paths.step5_rewritten.open(output_mode, encoding="utf-8") as fout, \
        paths.step5_done.open("a", encoding="utf-8") as fdone:

        for line in tqdm(fin, total=total, desc="LLM rewrite"):
            if not line.strip():
                continue

            raw_record = json.loads(line)
            idx = int(raw_record["idx"])

            if idx in done_idx:
                continue

            text = str(raw_record.get("text", "")).strip()
            if not text:
                fdone.write(str(idx) + "\n")
                done_idx.add(idx)
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
                )
                result_map = {int(result["idx"]): result for result in results}

                for raw in batch_raw:
                    row_idx = int(raw["idx"])
                    result = result_map[row_idx]

                    out = dict(raw)
                    out["rewritten"] = result.get("rewritten")
                    out["rewrite_is_medical"] = bool(result.get("is_medical", False))
                    out["rewrite_correction_made"] = bool(result.get("correction_made", False))
                    out["rewrite_reason"] = result.get("reason", "")

                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fdone.write(str(row_idx) + "\n")
                    done_idx.add(row_idx)

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
            )
            result_map = {int(result["idx"]): result for result in results}

            for raw in batch_raw:
                row_idx = int(raw["idx"])
                result = result_map[row_idx]

                out = dict(raw)
                out["rewritten"] = result.get("rewritten")
                out["rewrite_is_medical"] = bool(result.get("is_medical", False))
                out["rewrite_correction_made"] = bool(result.get("correction_made", False))
                out["rewrite_reason"] = result.get("reason", "")

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fdone.write(str(row_idx) + "\n")
                done_idx.add(row_idx)

            fout.flush()
            fdone.flush()

    LOGGER.info("Saved rewritten questions: %s", paths.step5_rewritten)


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
            "ablation": "global_gmm_no_icd_local_experts",
            "intended_difference_from_full_pipeline": (
                "Replace ICD-10 local GMM experts with one global L2 GMM."
            ),
            "paths": {k: str(v) for k, v in asdict(paths).items()},
            "config": asdict(config),
            "steps": steps,
            "device": str(device),
        },
        paths.run_config,
    )

    LOGGER.info("Ablation: No ICD Local Experts / Global L2 GMM")
    LOGGER.info("Output directory: %s", paths.output_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Steps: %s", steps)

    if "sample" in steps:
        step1_global_gmm_sampling(
            paths=paths,
            config=config,
            device=device,
            force_refit_gmm=args.force_refit_gmm,
            overwrite=args.overwrite,
        )

    if "invert" in steps:
        step2_vec2text_inversion(
            paths=paths,
            config=config,
            device=device,
            overwrite=args.overwrite,
        )

    if "reembed" in steps:
        step3_reembed_decoded_texts(
            paths=paths,
            config=config,
            device=device,
            overwrite=args.overwrite,
        )

    if "gibberish" in steps:
        step4_gibberish_detection(
            paths=paths,
            config=config,
            device=device,
            overwrite=args.overwrite,
        )

    if "rewrite" in steps:
        step5_llm_rewrite(
            paths=paths,
            config=config,
            api_key_env=args.api_key_env,
            base_url_env=args.base_url_env,
            base_url_arg=args.base_url,
            overwrite=args.overwrite,
        )

    LOGGER.info("Global GMM ablation completed. Results saved in %s", paths.output_dir)


if __name__ == "__main__":
    main()