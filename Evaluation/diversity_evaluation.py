#!/usr/bin/env python3
"""
evaluate_diversity.py

Monte Carlo subsampling diversity evaluation for generated medical questions.

Metrics:
  - Dist-1
  - Dist-2
  - Self-BLEU
  - V-EE
  - Topic Coverage (%)
  - Topic Entropy
  - Vendi Score
  - Gzip Ratio
  - Mean Pairwise Cosine Distance

The script is reviewer-friendly:
  - no local absolute paths;
  - no hard-coded identity-specific directories;
  - inputs are provided through a manifest file;
  - outputs are JSON cache + CSV table;
  - supports resume-style cache updates;
  - preserves existing cached metrics unless --force_recompute is passed.

Manifest format, CSV or JSONL:
  method,path
  MedQuAD+PubMedQA (GT),HUGGINGFACE_MIXED
  M1 GPT-5.4,outputs/baselines/m1__gpt-5.4.jsonl
  M2 GPT-5.4,outputs/baselines/m2__gpt-5.4.jsonl
  Ours-gpt-5.4-stratifiedL2,outputs/ours/step4_final_refined_gpt-5.4.jsonl

The ground-truth row must be named by --gt_name and may use path HUGGINGFACE_MIXED.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.util import ngrams
except ImportError as exc:
    raise ImportError(
        "This script requires nltk. Install it with: pip install nltk"
    ) from exc

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
    from sklearn.cluster import KMeans
except ImportError as exc:
    raise ImportError(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
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

TARGET_METRICS = [
    "Dist-1",
    "Dist-2",
    "Self-BLEU",
    "V-EE",
    "Topic Cov (%)",
    "Topic Ent",
    "Vendi Score",
    "Gzip Ratio",
    "Mean Pairwise Dist",
]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    path: str


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


def ensure_nltk() -> None:
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


# ---------------------------------------------------------------------
# Manifest / data loading
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
                f"Found columns: {df.columns.tolist()}"
            )
        return [
            MethodSpec(method=str(row["method"]).strip(), path=str(row["path"]).strip())
            for _, row in df.iterrows()
            if str(row["method"]).strip()
        ]

    if path.suffix.lower() == ".jsonl":
        specs: List[MethodSpec] = []
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


def load_huggingface_gt_texts() -> List[str]:
    if load_dataset is None:
        raise ImportError(
            "datasets is required to load HUGGINGFACE_MIXED ground truth. "
            "Install with: pip install datasets"
        )

    logging.info("Loading HuggingFace ground truth: MedQuAD + PubMedQA.")

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


def load_texts_with_duplicates(
    method_name: str,
    path: str,
    max_pool: int,
    seed: int,
) -> List[str]:
    """
    Load texts while preserving duplicates.

    If there are more than max_pool samples, globally sample without replacement
    using the provided seed.
    """
    if path == "HUGGINGFACE_MIXED" or method_name.lower().endswith("(gt)"):
        all_texts = load_huggingface_gt_texts()
    else:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found for {method_name}: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif suffix == ".txt":
            all_texts = [
                line.strip()
                for line in file_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            df = None
        else:
            raise ValueError(f"Unsupported file format for {file_path}: {suffix}")

        if suffix != ".txt":
            assert df is not None
            text_col = find_text_column(df, str(file_path), method_name)
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


# ---------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------

class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length

        logging.info("Loading embedding model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        raw_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = raw_model.encoder if hasattr(raw_model, "encoder") else raw_model
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        if len(texts) == 0:
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
# Diversity metrics
# ---------------------------------------------------------------------

def calc_distinct_n(texts: Sequence[str], n: int) -> float:
    all_ngrams: List[Tuple[str, ...]] = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        if len(tokens) < n:
            continue
        all_ngrams.extend(list(ngrams(tokens, n)))

    return float(len(set(all_ngrams)) / (len(all_ngrams) + 1e-12))


def calc_self_bleu(
    texts: Sequence[str],
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    tokenized_sents = [word_tokenize(t.lower()) for t in texts]

    if len(tokenized_sents) > sample_size:
        idx = rng.choice(len(tokenized_sents), sample_size, replace=False)
        tokenized_sents = [tokenized_sents[i] for i in idx.tolist()]

    if len(tokenized_sents) <= 1:
        return 0.0

    bleu_scores: List[float] = []
    smoothing = SmoothingFunction().method1

    for i, hyp in enumerate(tokenized_sents):
        refs = tokenized_sents[:i] + tokenized_sents[i + 1 :]
        if not refs:
            continue
        score = sentence_bleu(
            refs,
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
        bleu_scores.append(float(score))

    return float(np.mean(bleu_scores)) if bleu_scores else 0.0


def calc_vee_diversity(embeddings: np.ndarray) -> float:
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    probs = s / (np.sum(s) + 1e-12)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def build_topic_model(real_embeddings: np.ndarray, n_topics: int, seed: int) -> KMeans:
    kmeans = KMeans(n_clusters=n_topics, random_state=seed, n_init=5)
    kmeans.fit(real_embeddings)
    return kmeans


def calc_soft_coverage(
    generated_embeddings: np.ndarray,
    kmeans_model: KMeans,
) -> Tuple[float, float]:
    n_topics = kmeans_model.n_clusters
    labels = kmeans_model.predict(generated_embeddings)

    counts = np.bincount(labels, minlength=n_topics)
    coverage_ratio = np.sum(counts > 0) / n_topics

    topic_probs = counts / (np.sum(counts) + 1e-12)
    topic_entropy = -np.sum([p * np.log(p) for p in topic_probs if p > 0])

    return float(coverage_ratio * 100.0), float(topic_entropy)


def calc_vendi_score(embeddings: np.ndarray) -> float:
    embs_norm = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )

    kernel = embs_norm @ embs_norm.T
    kernel = kernel / kernel.shape[0]

    eigenvalues = np.linalg.eigvalsh(kernel)
    eigenvalues = eigenvalues[eigenvalues > 1e-9]

    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def calc_compression_ratio(texts: Sequence[str]) -> float:
    raw_text = " ".join(texts).encode("utf-8")
    if len(raw_text) == 0:
        return 0.0
    compressed = gzip.compress(raw_text)
    return float(len(compressed) / (len(raw_text) + 1e-12))


def calc_mean_pairwise_distance(embeddings: np.ndarray) -> float:
    embs_norm = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )

    sim = embs_norm @ embs_norm.T
    dist = 1.0 - sim

    n = dist.shape[0]
    if n <= 1:
        return 0.0

    upper_tri = np.triu_indices(n, k=1)
    return float(np.mean(dist[upper_tri]))


def compute_selected_metrics(
    texts: Sequence[str],
    embeddings: np.ndarray,
    topic_model: KMeans,
    metrics_to_compute: Sequence[str],
    self_bleu_sample_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    if "Dist-1" in metrics_to_compute:
        result["Dist-1"] = calc_distinct_n(texts, n=1)

    if "Dist-2" in metrics_to_compute:
        result["Dist-2"] = calc_distinct_n(texts, n=2)

    if "Self-BLEU" in metrics_to_compute:
        result["Self-BLEU"] = calc_self_bleu(
            texts,
            sample_size=self_bleu_sample_size,
            rng=rng,
        )

    if "V-EE" in metrics_to_compute:
        result["V-EE"] = calc_vee_diversity(embeddings)

    if "Topic Cov (%)" in metrics_to_compute or "Topic Ent" in metrics_to_compute:
        coverage, entropy = calc_soft_coverage(embeddings, topic_model)

        if "Topic Cov (%)" in metrics_to_compute:
            result["Topic Cov (%)"] = coverage

        if "Topic Ent" in metrics_to_compute:
            result["Topic Ent"] = entropy

    if "Vendi Score" in metrics_to_compute:
        result["Vendi Score"] = calc_vendi_score(embeddings)

    if "Gzip Ratio" in metrics_to_compute:
        result["Gzip Ratio"] = calc_compression_ratio(texts)

    if "Mean Pairwise Dist" in metrics_to_compute:
        result["Mean Pairwise Dist"] = calc_mean_pairwise_distance(embeddings)

    return result


# ---------------------------------------------------------------------
# Cache and formatting
# ---------------------------------------------------------------------

def csv_metric_key(metric_name: str) -> str:
    if metric_name == "Self-BLEU":
        return f"{metric_name} ↓"
    return f"{metric_name} ↑"


def format_metric(metric_name: str, mean_val: float, std_val: float) -> str:
    if metric_name == "Topic Cov (%)":
        return f"{mean_val:.1f} ± {std_val:.1f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def metric_missing_from_cache(entry: Optional[Dict[str, Any]], metric_name: str) -> bool:
    if not entry:
        return True

    csv_data = entry.get("csv_data", {})
    plot_data = entry.get("plot_data", {})

    return csv_metric_key(metric_name) not in csv_data or metric_name not in plot_data


def load_json_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_cache(cache: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def maybe_backup_cache(cache_path: Path, backup_path: Optional[Path]) -> None:
    if not cache_path.exists() or backup_path is None:
        return
    if backup_path.exists():
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, backup_path)
    logging.info("Created cache backup: %s", backup_path)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate_diversity(args: argparse.Namespace) -> None:
    ensure_nltk()
    set_seed(args.seed)

    manifest = load_manifest(Path(args.manifest))
    if not manifest:
        raise ValueError("Manifest is empty.")

    cache_path = Path(args.cache_file)
    csv_path = Path(args.csv_file)

    maybe_backup_cache(cache_path, Path(args.backup_file) if args.backup_file else None)
    evaluation_cache = load_json_cache(cache_path)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = TextEmbedder(
        model_name=args.embedding_model,
        device=device,
        max_length=args.max_length,
    )

    methods_texts: Dict[str, List[str]] = {}
    methods_embeddings: Dict[str, np.ndarray] = {}

    logging.info("Loading texts and computing embeddings.")

    for spec in manifest:
        logging.info("Loading method=%s", spec.method)

        try:
            texts = load_texts_with_duplicates(
                method_name=spec.method,
                path=spec.path,
                max_pool=args.pool_size,
                seed=args.seed,
            )
        except Exception as exc:
            logging.warning("Skipping %s due to loading error: %s", spec.method, repr(exc))
            continue

        if len(texts) < args.min_samples:
            logging.warning(
                "Skipping %s: only %d valid samples, below --min_samples=%d.",
                spec.method,
                len(texts),
                args.min_samples,
            )
            continue

        methods_texts[spec.method] = texts
        methods_embeddings[spec.method] = embedder.encode(texts, batch_size=args.batch_size)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.gt_name not in methods_embeddings:
        raise RuntimeError(
            f"Ground-truth method `{args.gt_name}` was not loaded successfully. "
            "It is required for KMeans topic-space fitting."
        )

    logging.info("Fitting ground-truth topic model with KMeans.")
    topic_model = build_topic_model(
        methods_embeddings[args.gt_name],
        n_topics=args.n_topics,
        seed=args.seed,
    )

    logging.info(
        "Running Monte Carlo subsampling evaluation: sample_size=%d, runs=%d.",
        args.sample_size,
        args.runs,
    )

    for method_name in progress(methods_embeddings.keys(), desc="Evaluating methods"):
        texts = np.array(methods_texts[method_name])
        embeddings = methods_embeddings[method_name]
        pool_len = len(texts)

        if pool_len < args.sample_size:
            logging.warning(
                "Skipping %s: pool_len=%d < sample_size=%d.",
                method_name,
                pool_len,
                args.sample_size,
            )
            continue

        existing_entry = evaluation_cache.get(method_name)

        if args.force_recompute or existing_entry is None:
            metrics_to_compute = TARGET_METRICS.copy()
            model_result: Dict[str, Any] = {"Model / Method": method_name}
            model_plot_data: Dict[str, Any] = {"Model": method_name}
        else:
            metrics_to_compute = [
                metric
                for metric in TARGET_METRICS
                if metric_missing_from_cache(existing_entry, metric)
            ]

            model_result = existing_entry.get("csv_data", {"Model / Method": method_name})
            model_plot_data = existing_entry.get("plot_data", {"Model": method_name})

            model_result.setdefault("Model / Method", method_name)
            model_plot_data.setdefault("Model", method_name)

        if not metrics_to_compute:
            logging.info("%s: all target metrics already cached. Skipping.", method_name)
            continue

        logging.info("%s: computing metrics %s", method_name, metrics_to_compute)

        run_scores: Dict[str, List[float]] = {metric: [] for metric in metrics_to_compute}
        rng = np.random.default_rng(args.seed)

        for _ in range(args.runs):
            idx = rng.choice(pool_len, size=args.sample_size, replace=False)
            sub_texts = texts[idx].tolist()
            sub_embeddings = embeddings[idx]

            metrics = compute_selected_metrics(
                texts=sub_texts,
                embeddings=sub_embeddings,
                topic_model=topic_model,
                metrics_to_compute=metrics_to_compute,
                self_bleu_sample_size=args.self_bleu_sample_size,
                rng=rng,
            )

            for metric, value in metrics.items():
                run_scores[metric].append(value)

        for metric in metrics_to_compute:
            arr = np.array(run_scores[metric], dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            model_result[csv_metric_key(metric)] = format_metric(metric, mean_val, std_val)
            model_plot_data[metric] = {
                "mean": mean_val,
                "err": std_val,
            }

        evaluation_cache[method_name] = {
            "csv_data": model_result,
            "plot_data": model_plot_data,
        }

        save_json_cache(evaluation_cache, cache_path)
        logging.info("Updated cache for %s: %s", method_name, cache_path)

    display_cols = ["Model / Method"] + [csv_metric_key(metric) for metric in TARGET_METRICS]
    rows: List[Dict[str, Any]] = []

    for method_name, entry in evaluation_cache.items():
        csv_data = entry.get("csv_data", {})
        row = {col: csv_data.get(col, "") for col in display_cols}
        row["Model / Method"] = csv_data.get("Model / Method", method_name)
        rows.append(row)

    df_results = pd.DataFrame(rows, columns=display_cols)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(csv_path, index=False)

    logging.info("Saved diversity cache: %s", cache_path)
    logging.info("Saved diversity CSV: %s", csv_path)

    print("\n" + "=" * 120)
    print(
        f"DIVERSITY LEADERBOARD "
        f"(N={args.sample_size}, {args.runs} runs, Mean ± Std)"
    )
    print("=" * 120)
    print(df_results.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo diversity evaluation for medical question generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--gt_name", type=str, default="MedQuAD+PubMedQA (GT)")

    parser.add_argument("--cache_file", type=str, default="outputs/diversity_cache.json")
    parser.add_argument("--backup_file", type=str, default="outputs/diversity_cache.backup.json")
    parser.add_argument("--csv_file", type=str, default="outputs/diversity_results.csv")

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/gtr-t5-base")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--pool_size", type=int, default=2000)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--self_bleu_sample_size", type=int, default=500)
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--min_samples", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    evaluate_diversity(args)


if __name__ == "__main__":
    main()