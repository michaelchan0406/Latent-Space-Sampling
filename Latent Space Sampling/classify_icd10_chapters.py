#!/usr/bin/env python3
"""
ICD-10 Chapter Classification for Medical Benchmark Questions.

This script is a preprocessing step for ICD-10-based local GMM training.
It classifies each medical question into exactly one ICD-10 chapter code
using an OpenAI-compatible chat-completions model.

Core logic preserved from the original notebook:
    1. Load questions from CSV, JSONL, or HuggingFace datasets.
    2. Deduplicate and optionally shuffle texts.
    3. Classify questions into ICD-10 chapters using JSON-mode LLM calls.
    4. Process data incrementally in chunks.
    5. Resume from existing labeled CSV files.
    6. Save one labeled CSV per source.
    7. Print and save a final processing summary.

Required packages:
    pip install openai pandas tqdm datasets

API key:
    export OPENAI_API_KEY="..."

Example source config, sources.json:
[
  {
    "name": "global_gmm_gpt5.4",
    "path": "./data/step5_llm_rewritten_questions.jsonl",
    "format": "jsonl",
    "text_column": "rewritten"
  },
  {
    "name": "baseline_self_instruct",
    "path": "./data/medical_questions.csv",
    "format": "csv",
    "text_column": "medical_question"
  },
  {
    "name": "MedQuAD+PubMedQA_GT",
    "format": "huggingface_mixed"
  }
]

Run:
    python classify_icd10_chapters.py \
        --sources-json ./sources.json \
        --output-dir ./artifacts/icd10_labeled \
        --model gpt-4o-mini \
        --batch-size 10 \
        --chunk-size 500 \
        --seed 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


ICD_CHAPTERS: List[Tuple[str, str]] = [
    ("A00-B99", "Certain infectious and parasitic diseases"),
    ("C00-D48", "Neoplasms"),
    ("D50-D89", "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
    ("E00-E90", "Endocrine, nutritional and metabolic diseases"),
    ("F00-F99", "Mental and behavioural disorders"),
    ("G00-G99", "Diseases of the nervous system"),
    ("H00-H59", "Diseases of the eye and adnexa"),
    ("H60-H95", "Diseases of the ear and mastoid process"),
    ("I00-I99", "Diseases of the circulatory system"),
    ("J00-J99", "Diseases of the respiratory system"),
    ("K00-K93", "Diseases of the digestive system"),
    ("L00-L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00-M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00-N99", "Diseases of the genitourinary system"),
    ("O00-O99", "Pregnancy, childbirth and the puerperium"),
    ("P00-P96", "Certain conditions originating in the perinatal period"),
    ("Q00-Q99", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("R00-R99", "Symptoms, signs and abnormal clinical and laboratory findings"),
    ("S00-T98", "Injury, poisoning and certain other consequences of external causes"),
    ("V01-Y98", "External causes of morbidity and mortality"),
    ("Z00-Z99", "Factors influencing health status and contact with health services"),
    ("U00-U99", "Codes for special purposes"),
]

VALID_CHAPTER_CODES: Set[str] = {code for code, _ in ICD_CHAPTERS}

DEFAULT_TEXT_COLUMNS: Tuple[str, ...] = (
    "rewritten",
    "medical_question",
    "question",
    "Question",
    "text",
    "doc",
    "prompt",
)


@dataclass(frozen=True)
class SourceSpec:
    """One input source to classify."""

    name: str
    format: str
    path: Optional[Path] = None
    text_column: Optional[str] = None


@dataclass(frozen=True)
class Config:
    """Runtime configuration."""

    sources_json: Path
    output_dir: Path
    model: str
    api_key_env: str
    base_url: Optional[str]
    seed: int
    batch_size: int
    chunk_size: int
    max_chars: int
    min_text_chars: int
    max_samples: Optional[int]
    request_timeout: float
    max_retries: int
    retry_sleep_seconds: float
    fallback_sleep_seconds: float
    shuffle: bool
    force: bool


def build_system_prompt() -> str:
    """Build the ICD-10 chapter classification system prompt."""
    chapter_str = "\n".join(f"- {code}: {desc}" for code, desc in ICD_CHAPTERS)

    return (
        "You are an expert ICD-10 classifier. "
        "Classify each medical question into exactly ONE ICD-10 chapter code.\n\n"
        f"Allowed ICD-10 chapters:\n{chapter_str}\n\n"
        "Rules:\n"
        "1. Return strictly a JSON object.\n"
        "2. The JSON object must contain exactly one top-level key named 'results'.\n"
        "3. 'results' must be a list of objects.\n"
        "4. Each object must contain 'id' and 'chapter_code'.\n"
        "5. 'chapter_code' must be one of the allowed ICD-10 chapter ranges above.\n"
        "6. Do not return explanations, markdown, or any extra text."
    )


def sanitize_name(name: str) -> str:
    """Make a source name safe for filenames."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "source"


def load_sources_config(path: Path) -> List[SourceSpec]:
    """Load source specifications from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Sources JSON does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Sources JSON must be a list of source objects.")

    sources: List[SourceSpec] = []

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Source entry {idx} must be an object.")

        name = str(item.get("name", "")).strip()
        source_format = str(item.get("format", "")).strip().lower()
        path_value = item.get("path")
        text_column = item.get("text_column")

        if not name:
            raise ValueError(f"Source entry {idx} is missing required field 'name'.")

        if source_format not in {"csv", "jsonl", "huggingface_mixed"}:
            raise ValueError(
                f"Source '{name}' has unsupported format '{source_format}'. "
                "Supported formats: csv, jsonl, huggingface_mixed."
            )

        source_path = Path(path_value) if path_value else None

        if source_format in {"csv", "jsonl"}:
            if source_path is None:
                raise ValueError(f"Source '{name}' requires a 'path'.")
            if not source_path.exists():
                raise FileNotFoundError(f"Source '{name}' path does not exist: {source_path}")

        sources.append(
            SourceSpec(
                name=name,
                format=source_format,
                path=source_path,
                text_column=str(text_column) if text_column else None,
            )
        )

    if not sources:
        raise ValueError("No sources found in sources JSON.")

    return sources


def select_text_column(columns: Sequence[str], requested: Optional[str]) -> str:
    """Select the text column from a table."""
    if requested:
        if requested not in columns:
            raise KeyError(
                f"Requested text column '{requested}' not found. "
                f"Available columns: {list(columns)}"
            )
        return requested

    for candidate in DEFAULT_TEXT_COLUMNS:
        if candidate in columns:
            return candidate

    raise KeyError(
        "Could not infer text column. "
        f"Available columns: {list(columns)}. "
        f"Please set text_column in sources.json."
    )


def deduplicate_texts(texts: Iterable[Any], *, min_text_chars: int) -> List[str]:
    """Clean, filter, and deduplicate texts while preserving first occurrence."""
    seen: Set[str] = set()
    cleaned: List[str] = []

    for value in texts:
        text = str(value).strip()

        if len(text) < min_text_chars:
            continue

        if text in seen:
            continue

        seen.add(text)
        cleaned.append(text)

    return cleaned


def load_csv_texts(path: Path, *, text_column: Optional[str], min_text_chars: int) -> List[str]:
    """Load source texts from a CSV file."""
    df = pd.read_csv(path)
    column = select_text_column(df.columns.tolist(), text_column)
    return deduplicate_texts(df[column].dropna().tolist(), min_text_chars=min_text_chars)


def load_jsonl_texts(path: Path, *, text_column: Optional[str], min_text_chars: int) -> List[str]:
    """
    Load source texts from a JSONL file.

    This streams the file line-by-line rather than loading the full file into memory.
    If text_column is omitted, the first matching column from DEFAULT_TEXT_COLUMNS is used
    independently for each JSON object.
    """
    texts: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] Skipping invalid JSON line {line_number} in {path.name}: {exc}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(obj, dict):
                continue

            if text_column:
                value = obj.get(text_column)
            else:
                value = None
                for candidate in DEFAULT_TEXT_COLUMNS:
                    if candidate in obj:
                        value = obj.get(candidate)
                        break

            if value is not None:
                texts.append(str(value))

    return deduplicate_texts(texts, min_text_chars=min_text_chars)


def load_huggingface_mixed_texts(*, min_text_chars: int) -> List[str]:
    """
    Load the MedQuAD + PubMedQA reference mixture used in the original notebook.

    Requires:
        pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading HuggingFace sources requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    print("   -> Loading MedQuAD from HuggingFace.")
    medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    medquad_texts = [
        row.get("Question", row.get("question", ""))
        for row in medquad
    ]

    print("   -> Loading PubMedQA unlabeled split from HuggingFace.")
    pubmedqa = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
    pubmedqa_texts = [
        row.get("question", "")
        for row in pubmedqa
    ]

    return deduplicate_texts(
        list(medquad_texts) + list(pubmedqa_texts),
        min_text_chars=min_text_chars,
    )


def load_source_texts(source: SourceSpec, *, min_text_chars: int) -> List[str]:
    """Load texts for a configured source."""
    if source.format == "csv":
        assert source.path is not None
        return load_csv_texts(
            source.path,
            text_column=source.text_column,
            min_text_chars=min_text_chars,
        )

    if source.format == "jsonl":
        assert source.path is not None
        return load_jsonl_texts(
            source.path,
            text_column=source.text_column,
            min_text_chars=min_text_chars,
        )

    if source.format == "huggingface_mixed":
        return load_huggingface_mixed_texts(min_text_chars=min_text_chars)

    raise ValueError(f"Unsupported source format: {source.format}")


def read_existing_labeled_texts(save_path: Path) -> Set[str]:
    """Read existing labeled CSV and return already processed texts."""
    if not save_path.exists():
        return set()

    try:
        df = pd.read_csv(save_path)
    except Exception as exc:
        print(
            f"[WARN] Could not read existing file {save_path}: {exc}. "
            "Proceeding as if no items were processed.",
            file=sys.stderr,
        )
        return set()

    if "text" not in df.columns:
        print(
            f"[WARN] Existing file {save_path} has no 'text' column. "
            "Proceeding as if no items were processed.",
            file=sys.stderr,
        )
        return set()

    return set(df["text"].dropna().astype(str).tolist())


def create_client(*, api_key_env: str, base_url: Optional[str], timeout: float) -> OpenAI:
    """Create an OpenAI-compatible client from environment variables."""
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set environment variable {api_key_env} before running."
        )

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    return OpenAI(api_key=api_key, timeout=timeout)


def parse_llm_results(raw_text: str, subset: Sequence[str]) -> Optional[List[Dict[str, str]]]:
    """Parse and validate one JSON-mode LLM response."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    items = parsed.get("results")
    if not isinstance(items, list):
        return None

    output: List[Dict[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        idx = item.get("id")
        chapter_code = item.get("chapter_code")

        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue

        if not (0 <= idx_int < len(subset)):
            continue

        if not isinstance(chapter_code, str):
            continue

        chapter_code = chapter_code.strip()

        if chapter_code not in VALID_CHAPTER_CODES:
            print(
                f"[WARN] Invalid ICD-10 chapter code returned: {chapter_code}",
                file=sys.stderr,
            )
            continue

        output.append(
            {
                "text": subset[idx_int],
                "chapter_code": chapter_code,
            }
        )

    return output


def classify_subset_once(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    subset: Sequence[str],
    max_chars: int,
) -> Optional[List[Dict[str, str]]]:
    """Classify one subset of texts with a single API request."""
    payload = [
        {
            "id": idx,
            "text": text[:max_chars],
        }
        for idx, text in enumerate(subset)
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    return parse_llm_results(raw, subset)


def classify_subset_with_retries(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    subset: Sequence[str],
    max_chars: int,
    max_retries: int,
    retry_sleep_seconds: float,
) -> Optional[List[Dict[str, str]]]:
    """Classify a subset with retry handling."""
    for attempt in range(1, max_retries + 1):
        try:
            result = classify_subset_once(
                client=client,
                model=model,
                system_prompt=system_prompt,
                subset=subset,
                max_chars=max_chars,
            )

            if result is not None:
                return result

            raise ValueError("Model response could not be parsed into valid ICD-10 results.")

        except Exception as exc:
            is_final = attempt == max_retries
            print(
                f"[WARN] Classification call failed "
                f"(attempt {attempt}/{max_retries}, batch_size={len(subset)}): {exc}",
                file=sys.stderr,
            )

            if is_final:
                return None

            time.sleep(retry_sleep_seconds * attempt)

    return None


def classify_texts(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    texts: Sequence[str],
    batch_size: int,
    max_chars: int,
    max_retries: int,
    retry_sleep_seconds: float,
    fallback_sleep_seconds: float,
) -> List[Dict[str, str]]:
    """
    Classify texts into ICD-10 chapters.

    If a batch fails, the function falls back to individual processing, matching
    the original notebook's degradation behavior.
    """
    results: List[Dict[str, str]] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch = list(texts[start : start + batch_size])

        batch_results = classify_subset_with_retries(
            client=client,
            model=model,
            system_prompt=system_prompt,
            subset=batch,
            max_chars=max_chars,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )

        if batch_results is not None:
            results.extend(batch_results)
            continue

        print(
            f"[WARN] Batch failed; falling back to individual processing "
            f"for {len(batch)} item(s).",
            file=sys.stderr,
        )

        for single_text in batch:
            time.sleep(fallback_sleep_seconds)

            single_result = classify_subset_with_retries(
                client=client,
                model=model,
                system_prompt=system_prompt,
                subset=[single_text],
                max_chars=max_chars,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )

            if single_result:
                results.extend(single_result)

    return results


def append_labeled_csv(save_path: Path, rows: Sequence[Mapping[str, str]], *, source_name: str) -> None:
    """Append labeled rows to a CSV file."""
    if not rows:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not save_path.exists()

    fieldnames = ["source", "text", "chapter_code"]

    with save_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "source": source_name,
                    "text": row["text"],
                    "chapter_code": row["chapter_code"],
                }
            )


def process_source(
    *,
    source: SourceSpec,
    config: Config,
    client: OpenAI,
    system_prompt: str,
) -> Dict[str, Any]:
    """Process one source and return a summary record."""
    safe_name = sanitize_name(source.name)
    save_path = config.output_dir / f"labeled_{safe_name}.csv"

    print(f"\n>>> Processing source: {source.name}")
    print(f"   -> Output: {save_path}")

    all_texts = load_source_texts(source, min_text_chars=config.min_text_chars)

    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(all_texts)

    if config.max_samples is not None:
        all_texts = all_texts[: config.max_samples]

    existing_texts = set() if config.force else read_existing_labeled_texts(save_path)
    remaining = [text for text in all_texts if text not in existing_texts]

    print(f"   -> Total available texts: {len(all_texts)}")
    print(f"   -> Already processed: {len(existing_texts)}")
    print(f"   -> Remaining to classify: {len(remaining)}")

    if not remaining:
        return {
            "source": source.name,
            "output_file": str(save_path),
            "processed": len(existing_texts),
            "total": len(all_texts),
            "newly_processed": 0,
            "status": "complete",
        }

    newly_processed = 0

    for start in range(0, len(remaining), config.chunk_size):
        chunk = remaining[start : start + config.chunk_size]
        chunk_index = start // config.chunk_size + 1

        print(f"\n   --- Chunk {chunk_index} | size={len(chunk)} ---")

        labeled_rows = classify_texts(
            client=client,
            model=config.model,
            system_prompt=system_prompt,
            texts=chunk,
            batch_size=config.batch_size,
            max_chars=config.max_chars,
            max_retries=config.max_retries,
            retry_sleep_seconds=config.retry_sleep_seconds,
            fallback_sleep_seconds=config.fallback_sleep_seconds,
        )

        if labeled_rows:
            append_labeled_csv(save_path, labeled_rows, source_name=source.name)
            newly_processed += len(labeled_rows)
            print(f"   -> [SAVE] Wrote {len(labeled_rows)} labeled rows.")
        else:
            print("   -> [WARN] No labels recovered for this chunk.", file=sys.stderr)

    total_processed = len(read_existing_labeled_texts(save_path))
    status = "complete" if total_processed >= len(all_texts) and len(all_texts) > 0 else "partial"

    return {
        "source": source.name,
        "output_file": str(save_path),
        "processed": total_processed,
        "total": len(all_texts),
        "newly_processed": newly_processed,
        "status": status,
    }


def save_summary(output_dir: Path, summary: Sequence[Mapping[str, Any]]) -> Path:
    """Save final processing summary as CSV."""
    output_path = output_dir / "icd10_classification_summary.csv"
    pd.DataFrame(summary).to_csv(output_path, index=False)
    return output_path


def print_summary(summary: Sequence[Mapping[str, Any]]) -> None:
    """Print final processing summary."""
    if not summary:
        print("[WARN] No summary records to print.")
        return

    df = pd.DataFrame(summary)

    print("\n" + "=" * 80)
    print(f"{'ICD-10 CLASSIFICATION SUMMARY':^80}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify medical benchmark questions into ICD-10 chapters."
    )

    parser.add_argument(
        "--sources-json",
        type=Path,
        required=True,
        help="JSON file containing input source specifications.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where labeled ICD-10 CSV files will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI-compatible model for ICD-10 classification. Default: gpt-4o-mini.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing API key. Default: OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used when shuffling sources. Default: 2026.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions sent per LLM request. Default: 10.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of remaining questions processed before writing progress. Default: 500.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=256,
        help="Maximum characters from each question sent to the classifier. Default: 256.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=10,
        help="Minimum text length after stripping. Default: 10.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per source, useful for debugging. Default: no cap.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="API request timeout in seconds. Default: 60.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per API request. Default: 3.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Base sleep between retries; multiplied by attempt index. Default: 2.",
    )
    parser.add_argument(
        "--fallback-sleep-seconds",
        type=float,
        default=0.1,
        help="Sleep before individual fallback calls. Default: 0.1.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable deterministic shuffling before classification.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing labeled CSV files and reprocess from scratch.",
    )

    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    if args.max_chars <= 0:
        raise ValueError("--max-chars must be positive.")

    if args.min_text_chars < 0:
        raise ValueError("--min-text-chars must be non-negative.")

    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")

    return Config(
        sources_json=args.sources_json,
        output_dir=args.output_dir,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        seed=args.seed,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_chars=args.max_chars,
        min_text_chars=args.min_text_chars,
        max_samples=args.max_samples,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        fallback_sleep_seconds=args.fallback_sleep_seconds,
        shuffle=not args.no_shuffle,
        force=args.force,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ICD-10 chapter classification for all configured sources."""
    config = parse_args(argv)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources_config(config.sources_json)

    client = create_client(
        api_key_env=config.api_key_env,
        base_url=config.base_url,
        timeout=config.request_timeout,
    )

    system_prompt = build_system_prompt()

    print("[INFO] Starting ICD-10 chapter classification.")
    print(f"[INFO] Sources config: {config.sources_json}")
    print(f"[INFO] Output directory: {config.output_dir}")
    print(f"[INFO] Model: {config.model}")
    print(f"[INFO] Number of sources: {len(sources)}")

    summary: List[Dict[str, Any]] = []

    for source in sources:
        try:
            record = process_source(
                source=source,
                config=config,
                client=client,
                system_prompt=system_prompt,
            )
            summary.append(record)

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopped by user.", file=sys.stderr)
            raise

        except Exception as exc:
            print(
                f"[ERROR] Source failed: {source.name}: {exc}",
                file=sys.stderr,
            )
            summary.append(
                {
                    "source": source.name,
                    "output_file": "",
                    "processed": 0,
                    "total": 0,
                    "newly_processed": 0,
                    "status": f"failed: {exc}",
                }
            )

    summary_path = save_summary(config.output_dir, summary)
    print_summary(summary)
    print(f"[SAVE] Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)