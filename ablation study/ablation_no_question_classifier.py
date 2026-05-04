#!/usr/bin/env python3
"""
Reviewer-friendly ablation pipeline: ICD-10 local GMM sampling without question classifier.

This script implements the "No Question Classifier" ablation.

Compared with the full pipeline, the intended experimental change is:
  - Full pipeline:
      ICD-10-based proportional local GMM sampling
      + latent question-form classifier filtering
      + vec2text decoding
      + gibberish filtering
      + LLM clinical rewriting
  - This ablation:
      ICD-10-based proportional local GMM sampling
      WITHOUT latent question-form classifier filtering
      + vec2text decoding
      + gibberish filtering
      + LLM clinical rewriting

This ablation tests whether the latent question-form classifier is necessary for
generating question-like decoded medical benchmark items.

No private paths, names, institutions, or API keys are embedded in this file.
Supply API credentials through environment variables.

Example:
  python ablation_no_question_classifier.py \\
    --data-root ./data \\
    --output-dir ./outputs/ablation_no_question_classifier \\
    --steps all

Expected default input layout:
  data/
    experts/
      gmms/
        gmm_<cluster_name>.joblib
    references/
      labeled_MedQuAD_PubMedQA_GT.csv

Outputs:
  output-dir/
    run_config.json
    step1_vectors_proportional_no_question_filter.npy
    step1_vectors_proportional_no_question_filter_meta.jsonl
    step1_sampling_quotas.json
    step2_texts.jsonl
    step3_clean_texts.jsonl
    step4_final_refined.jsonl
    step4_refined.done.txt
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

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

    step1_vectors: Path
    step1_meta: Path
    step1_quotas: Path

    step2_texts: Path
    step3_clean_texts: Path

    step4_refined: Path
    step4_done: Path

    run_config: Path


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    target_total: int
    sample_batch_size: int
    embedding_dim: int
    normalize_samples: bool

    vec2text_corrector: str
    vec2text_batch_size: int
    vec2text_steps: int
    vec2text_beam: int

    gibberish_model: str
    gibberish_batch_size: int
    gibberish_threshold: float

    rewrite_model: str
    rewrite_batch_size: int
    rewrite_temperature: float
    rewrite_timeout: float
    rewrite_max_retries: int
    sleep_between_retries: float

    ablation: str = "no_question_classifier"
    intended_difference_from_full_pipeline: str = (
        "Use the same ICD-10 proportional local GMM sampling as the full pipeline, "
        "but remove the latent question-form classifier."
    )


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
        description="No-question-classifier ablation for ICD-10 local GMM sampling."
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
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", "sample", "invert", "gibberish", "rewrite"],
        help="Pipeline steps to run.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-total", type=int, default=21000)
    parser.add_argument("--sample-batch-size", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument(
        "--no-normalize-samples",
        action="store_true",
        help="Disable L2 normalization after sampling from each local GMM.",
    )

    parser.add_argument("--vec2text-corrector", type=str, default="gtr-base")
    parser.add_argument("--vec2text-batch-size", type=int, default=40)
    parser.add_argument("--vec2text-steps", type=int, default=40)
    parser.add_argument("--vec2text-beam", type=int, default=4)

    parser.add_argument(
        "--disable-gibberish-filter",
        action="store_true",
        help="Skip gibberish filtering and copy Step 2 texts directly to Step 3 output.",
    )
    parser.add_argument(
        "--gibberish-model",
        type=str,
        default="madhurjindal/autonlp-Gibberish-Detector-492513457",
    )
    parser.add_argument("--gibberish-batch-size", type=int, default=64)
    parser.add_argument("--gibberish-threshold", type=float, default=0.90)

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

    return PipelinePaths(
        data_root=data_root,
        output_dir=output_dir,
        gmm_dir=gmm_dir,
        reference_csv=reference_csv,
        step1_vectors=output_dir / "step1_vectors_proportional_no_question_filter.npy",
        step1_meta=output_dir / "step1_vectors_proportional_no_question_filter_meta.jsonl",
        step1_quotas=output_dir / "step1_sampling_quotas.json",
        step2_texts=output_dir / "step2_texts.jsonl",
        step3_clean_texts=output_dir / "step3_clean_texts.jsonl",
        step4_refined=output_dir / "step4_final_refined.jsonl",
        step4_done=output_dir / "step4_refined.done.txt",
        run_config=output_dir / "run_config.json",
    )


def make_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        seed=args.seed,
        target_total=args.target_total,
        sample_batch_size=args.sample_batch_size,
        embedding_dim=args.embedding_dim,
        normalize_samples=not args.no_normalize_samples,
        vec2text_corrector=args.vec2text_corrector,
        vec2text_batch_size=args.vec2text_batch_size,
        vec2text_steps=args.vec2text_steps,
        vec2text_beam=args.vec2text_beam,
        gibberish_model=args.gibberish_model,
        gibberish_batch_size=args.gibberish_batch_size,
        gibberish_threshold=args.gibberish_threshold,
        rewrite_model=args.rewrite_model,
        rewrite_batch_size=args.rewrite_batch_size,
        rewrite_temperature=args.rewrite_temperature,
        rewrite_timeout=args.rewrite_timeout,
        rewrite_max_retries=args.rewrite_max_retries,
        sleep_between_retries=args.sleep_between_retries,
    )


def normalize_steps(raw_steps: Sequence[str]) -> List[str]:
    if "all" in raw_steps:
        return ["sample", "invert", "gibberish", "rewrite"]

    ordered = ["sample", "invert", "gibberish", "rewrite"]
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
        if not stem.startswith("gmm_"):
            continue
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
        "ablation": "no_question_classifier",
        "sampling": "ICD-10 proportional stratified local GMM sampling",
        "latent_question_classifier": "disabled",
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


def step1_stratified_sampling_without_question_filter(
    paths: PipelinePaths,
    config: PipelineConfig,
    overwrite: bool,
) -> np.ndarray:
    """
    Step 1: ICD-10 proportional local GMM sampling without question classifier.

    This is the only intended experimental change from the full pipeline:
      - The same ICD-10 super-cluster proportional quotas are used.
      - The same local GMM experts are used.
      - Sampled vectors are L2-normalized by default.
      - No latent question-form classifier is loaded or applied.
    """
    LOGGER.info("Step 1: ICD-10 proportional local GMM sampling WITHOUT question classifier")

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

    require_dir(paths.gmm_dir, "GMM directory")
    require_file(paths.reference_csv, "reference CSV")

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
        pbar = tqdm(total=target_n, desc=f"Cluster {cluster_name}")

        while accepted < target_n:
            current_batch_size = min(config.sample_batch_size, target_n - accepted)
            x_sampled, component_ids = gmm.sample(current_batch_size)
            x_sampled = x_sampled.astype(np.float32)

            if config.normalize_samples:
                x_sampled = l2_normalize(x_sampled)

            for vector, component_id in zip(x_sampled, component_ids):
                final_vectors.append(vector.astype(np.float32))
                final_meta.append(
                    {
                        "idx": len(final_meta),
                        "source_cluster": cluster_name,
                        "target_quota": int(target_n),
                        "source": "icd10_local_gmm_no_question_filter",
                        "gmm_component": int(component_id),
                    }
                )
                accepted += 1
                pbar.update(1)

        pbar.close()

        del gmm
        gc.collect()

    if not final_vectors:
        raise RuntimeError("No vectors were generated. Check local GMM paths and quotas.")

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
    """Step 2: Decode sampled vectors into raw text using vec2text."""
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


def copy_step2_to_step3(paths: PipelinePaths, overwrite: bool) -> None:
    require_file(paths.step2_texts, "Step 2 text output")

    if paths.step3_clean_texts.exists() and not overwrite:
        LOGGER.info("Step 3 output already exists: %s", paths.step3_clean_texts)
        return

    if paths.step3_clean_texts.exists() and overwrite:
        paths.step3_clean_texts.unlink()

    shutil.copyfile(paths.step2_texts, paths.step3_clean_texts)
    LOGGER.info("Copied Step 2 texts to Step 3 output: %s", paths.step3_clean_texts)


def step3_gibberish_filtering(
    paths: PipelinePaths,
    config: PipelineConfig,
    device: torch.device,
    overwrite: bool,
) -> None:
    """Step 3: Text-based gibberish filtering."""
    LOGGER.info("Step 3: gibberish filtering")

    require_file(paths.step2_texts, "Step 2 text output")

    if paths.step3_clean_texts.exists() and not overwrite:
        LOGGER.info("Step 3 output already exists: %s", paths.step3_clean_texts)
        return

    if paths.step3_clean_texts.exists() and overwrite:
        paths.step3_clean_texts.unlink()

    rows = read_jsonl(paths.step2_texts)
    if not rows:
        raise ValueError(f"No decoded texts found in {paths.step2_texts}")

    texts = [str(row.get("text", "")) for row in rows]

    detector = pipeline(
        "text-classification",
        model=config.gibberish_model,
        device=0 if device.type == "cuda" else -1,
    )

    clean_probe = "What is the recommended treatment for hypertension?"
    clean_label = detector(clean_probe)[0]["label"]
    LOGGER.info("Inferred clean gibberish-detector label: %s", clean_label)

    kept_count = 0

    for start in tqdm(
        range(0, len(texts), config.gibberish_batch_size),
        desc="Gibberish check",
    ):
        end = min(start + config.gibberish_batch_size, len(texts))
        batch = texts[start:end]
        results = detector(batch, truncation=True)

        out_rows: List[Dict[str, Any]] = []
        for offset, result in enumerate(results):
            source_row = rows[start + offset]
            is_clean = (
                result["label"] == clean_label
                and float(result["score"]) >= config.gibberish_threshold
            )

            if is_clean:
                out_rows.append(
                    {
                        "idx": int(source_row["idx"]),
                        "text": batch[offset],
                        "gib_label": str(result["label"]),
                        "gib_score": float(result["score"]),
                    }
                )

        if out_rows:
            append_jsonl(paths.step3_clean_texts, out_rows)
            kept_count += len(out_rows)

    LOGGER.info(
        "Saved Step 3 clean texts: %s; kept %d/%d",
        paths.step3_clean_texts,
        kept_count,
        len(texts),
    )


def make_openai_client(
    api_key_env: str,
    base_url: Optional[str],
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


def step4_llm_rewrite(
    paths: PipelinePaths,
    config: PipelineConfig,
    api_key_env: str,
    base_url: Optional[str],
    overwrite: bool,
) -> None:
    """Step 4: LLM-based clinical rewriting and validation."""
    LOGGER.info("Step 4: LLM-based clinical rewrite")

    require_file(paths.step3_clean_texts, "Step 3 clean text file")

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

    total = count_jsonl(paths.step3_clean_texts)
    LOGGER.info("Rewrite input: %s", paths.step3_clean_texts)
    LOGGER.info("Rewrite output: %s", paths.step4_refined)
    LOGGER.info("Already done: %d", len(done_ids))
    LOGGER.info("Rewrite model: %s", config.rewrite_model)

    batch_in: List[Dict[str, Any]] = []
    batch_raw: List[Dict[str, Any]] = []

    output_mode = "a" if paths.step4_refined.exists() else "w"

    with paths.step3_clean_texts.open("r", encoding="utf-8") as fin, \
        paths.step4_refined.open(output_mode, encoding="utf-8") as fout, \
        paths.step4_done.open("a", encoding="utf-8") as fdone:

        for line in tqdm(fin, total=total, desc="LLM rewrite"):
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

                    out = dict(raw)
                    out["rewritten"] = result.get("rewritten")
                    out["rewrite_is_medical"] = bool(result.get("is_medical", False))
                    out["rewrite_correction_made"] = bool(result.get("correction_made", False))
                    out["rewrite_reason"] = result.get("reason", "")

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

                out = dict(raw)
                out["rewritten"] = result.get("rewritten")
                out["rewrite_is_medical"] = bool(result.get("is_medical", False))
                out["rewrite_correction_made"] = bool(result.get("correction_made", False))
                out["rewrite_reason"] = result.get("reason", "")

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fdone.write(str(row_idx) + "\n")
                done_ids.add(row_idx)

            fout.flush()
            fdone.flush()

    LOGGER.info("Saved rewritten questions: %s", paths.step4_refined)


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
        },
        paths.run_config,
    )

    LOGGER.info("Ablation: ICD-10 local GMM sampling WITHOUT question classifier")
    LOGGER.info("Output directory: %s", paths.output_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Steps: %s", steps)

    if "sample" in steps:
        step1_stratified_sampling_without_question_filter(
            paths=paths,
            config=config,
            overwrite=args.overwrite,
        )

    if "invert" in steps:
        step2_vec2text_inversion(
            paths=paths,
            config=config,
            device=device,
            overwrite=args.overwrite,
        )

    if "gibberish" in steps:
        if args.disable_gibberish_filter:
            LOGGER.info("Gibberish filtering disabled by CLI flag.")
            copy_step2_to_step3(paths=paths, overwrite=args.overwrite)
        else:
            step3_gibberish_filtering(
                paths=paths,
                config=config,
                device=device,
                overwrite=args.overwrite,
            )

    if "rewrite" in steps:
        step4_llm_rewrite(
            paths=paths,
            config=config,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            overwrite=args.overwrite,
        )

    LOGGER.info(
        "No-question-classifier ablation completed. Results saved in %s",
        paths.output_dir,
    )


if __name__ == "__main__":
    main()