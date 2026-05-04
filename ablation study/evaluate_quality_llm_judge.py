#!/usr/bin/env python3
"""
LLM-as-a-judge quality evaluation for synthetic medical benchmark questions.

This script evaluates generated medical questions using an LLM judge on three
criteria:
  1. Fluency
  2. Clinical Validity
  3. Answerability

Each criterion is scored from 1 to 5.

This script is intended for evaluating outputs from the full pipeline and
ablation studies, including:
  - full pipeline
  - global GMM ablation
  - no question classifier ablation
  - no gibberish detector ablation

No API keys, private paths, names, institutions, or local drive paths are
embedded in this file. Supply credentials through environment variables.

Examples:

  Evaluate final rewritten questions:
    python evaluate_quality_llm_judge.py \\
      --input-file ./outputs/full_pipeline/step4_final_refined.jsonl \\
      --output-dir ./eval/full_pipeline \\
      --text-field rewritten \\
      --run-name full_pipeline

  Evaluate no-gibberish ablation final rewritten questions:
    python evaluate_quality_llm_judge.py \\
      --input-file ./outputs/ablation_no_gibberish/step4_final_refined_no_gibberish.jsonl \\
      --output-dir ./eval/ablation_no_gibberish \\
      --text-field rewritten \\
      --run-name no_gibberish

  Evaluate raw decoded or gibberish-filtered texts:
    python evaluate_quality_llm_judge.py \\
      --input-file ./outputs/full_pipeline/step3_clean_texts.jsonl \\
      --output-dir ./eval/full_pipeline_raw_clean_texts \\
      --text-field text \\
      --run-name full_pipeline_raw_clean_texts

Outputs:
  output-dir/
    <run-name>_judge_results.jsonl
    <run-name>_judge_results.csv
    <run-name>_judge_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


EVAL_SYSTEM_PROMPT = """
You are an expert medical professional and a strict linguistic evaluator.

Your task is to evaluate a generated medical question based on three criteria.
Score each criterion on a scale of 1 to 5, where 1 is worst and 5 is best.

Criteria:
1. Fluency:
   Is the grammar correct and natural?
   1 = gibberish or ungrammatical.
   5 = fluent, natural, native-level English.

2. Clinical Validity:
   Does the question make medical sense?
   Are the medical entities, conditions, interventions, and relationships used logically?
   1 = medically nonsensical or internally contradictory.
   5 = clinically realistic and medically coherent.

3. Answerability:
   Is the question specific enough to be answered using medical literature?
   1 = too vague, malformed, or impossible to answer.
   5 = specific, clear, and answerable.

Output your evaluation strictly as a JSON object in this exact schema:
{
  "Fluency": <int>,
  "Clinical_Validity": <int>,
  "Answerability": <int>,
  "Reasoning": "<A brief one-sentence explanation of your scores>"
}
"""


@dataclass(frozen=True)
class EvalConfig:
    input_file: str
    output_dir: str
    run_name: str
    text_field: str
    id_field: str
    sample_size: int
    seed: int
    judge_model: str
    temperature: float
    max_retries: int
    retry_sleep_seconds: float
    request_sleep_seconds: float
    api_key_env: str
    base_url: Optional[str]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated medical questions with an LLM judge."
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input JSONL or CSV file containing generated questions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation results will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name used in output filenames. Default: input filename stem.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="auto",
        help=(
            "Field containing the question text. Use 'rewritten' for final LLM-refined "
            "outputs, 'text' for raw decoded/gibberish-filtered text, or 'auto' to "
            "prefer rewritten > question > text."
        ),
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="idx",
        help="Optional field used as stable source ID. Default: idx.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of questions to sample. Use 0 or negative to evaluate all valid rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI-compatible model used as the judge.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge model temperature. Keep 0.0 for deterministic scoring.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retries per question.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Base sleep time between retries.",
    )
    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=0.5,
        help="Sleep time between successful requests to reduce rate-limit risk.",
    )

    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--base-url-env",
        type=str,
        default="OPENAI_BASE_URL",
        help="Environment variable containing optional base URL.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation outputs instead of resuming.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def make_config(args: argparse.Namespace) -> EvalConfig:
    run_name = args.run_name or args.input_file.stem
    base_url = args.base_url or os.environ.get(args.base_url_env)

    return EvalConfig(
        input_file=str(args.input_file.expanduser().resolve()),
        output_dir=str(args.output_dir.expanduser().resolve()),
        run_name=run_name,
        text_field=args.text_field,
        id_field=args.id_field,
        sample_size=args.sample_size,
        seed=args.seed,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        request_sleep_seconds=args.request_sleep_seconds,
        api_key_env=args.api_key_env,
        base_url=base_url,
    )


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_input_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL or CSV records into a list of dictionaries."""
    require_file(path, "input file")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [row for row in obj if isinstance(row, dict)]
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            return [row for row in obj["results"] if isinstance(row, dict)]
        raise ValueError(
            "JSON input must be either a list of records or an object with a 'results' list."
        )

    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {path.suffix}. Use .jsonl, .json, or .csv.")


def normalize_text_value(value: Any) -> Optional[str]:
    """Convert a possible text value into a valid non-empty string."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.lower() in {"none", "null", "nan"}:
        return None

    return text


def select_question_text(record: Dict[str, Any], text_field: str) -> Optional[str]:
    """
    Select the question text from a record.

    text_field='auto' prefers final rewritten outputs, then common alternatives.
    """
    if text_field != "auto":
        return normalize_text_value(record.get(text_field))

    for candidate_field in ("rewritten", "question", "Question", "text", "prompt"):
        text = normalize_text_value(record.get(candidate_field))
        if text is not None:
            return text

    return None


def prepare_eval_items(
    records: Sequence[Dict[str, Any]],
    text_field: str,
    id_field: str,
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Extract valid question texts and sample them reproducibly."""
    valid_items: List[Dict[str, Any]] = []

    for row_number, record in enumerate(records):
        question = select_question_text(record, text_field)
        if question is None:
            continue

        source_id = record.get(id_field, row_number)
        try:
            source_id = int(source_id)
        except Exception:
            source_id = str(source_id)

        valid_items.append(
            {
                "source_row": row_number,
                "source_id": source_id,
                "question": question,
            }
        )

    if not valid_items:
        raise ValueError(
            f"No valid question text found using text_field='{text_field}'."
        )

    rng = random.Random(seed)
    rng.shuffle(valid_items)

    if sample_size > 0:
        valid_items = valid_items[: min(sample_size, len(valid_items))]

    valid_items = sorted(valid_items, key=lambda item: item["source_row"])
    return valid_items


def make_openai_client(api_key_env: str, base_url: Optional[str]) -> Any:
    if OpenAI is None:
        raise ImportError("The openai package is required. Install it before running evaluation.")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set environment variable {api_key_env} before running evaluation."
        )

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response."""
    text = (text or "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response.")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Extracted judge response is not a JSON object.")

    return obj


def coerce_score(value: Any, field_name: str) -> int:
    """Convert a model-provided score to an integer in [1, 5]."""
    try:
        score = int(value)
    except Exception as exc:
        raise ValueError(f"Invalid score for {field_name}: {value}") from exc

    if score < 1 or score > 5:
        raise ValueError(f"Score for {field_name} is outside [1, 5]: {score}")

    return score


def normalize_judge_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize judge response schema."""
    return {
        "Fluency": coerce_score(response.get("Fluency"), "Fluency"),
        "Clinical_Validity": coerce_score(
            response.get("Clinical_Validity"),
            "Clinical_Validity",
        ),
        "Answerability": coerce_score(response.get("Answerability"), "Answerability"),
        "Reasoning": str(response.get("Reasoning", "")).strip(),
    }


def evaluate_question_via_api(
    client: Any,
    question_text: str,
    model: str,
    temperature: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> Dict[str, Any]:
    """Evaluate one question using an OpenAI-compatible chat-completions API."""
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Please evaluate the following generated medical question:\n\n"
                f"{question_text}"
            ),
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
            except Exception as json_mode_error:
                LOGGER.warning(
                    "JSON response_format failed; falling back without JSON mode: %s",
                    json_mode_error,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

            content = response.choices[0].message.content or ""
            parsed = extract_json_object(content)
            return normalize_judge_response(parsed)

        except Exception as exc:
            if attempt == max_retries:
                raise

            sleep_seconds = retry_sleep_seconds * attempt
            LOGGER.warning(
                "Judge call failed on attempt %d/%d: %s. Sleeping %.1fs.",
                attempt,
                max_retries,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable evaluation retry state.")


def load_completed_source_rows(results_jsonl: Path) -> Set[int]:
    """Load completed source_row IDs from a previous JSONL result file."""
    completed: Set[int] = set()

    if not results_jsonl.exists():
        return completed

    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(int(row["source_row"]))
            except Exception:
                continue

    return completed


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_summary(df: pd.DataFrame, config: EvalConfig) -> Dict[str, Any]:
    """Compute aggregate quality metrics."""
    if df.empty:
        raise ValueError("No evaluation results available for summary.")

    score_fields = ["Fluency", "Clinical_Validity", "Answerability"]

    summary: Dict[str, Any] = {
        "run_name": config.run_name,
        "input_file": config.input_file,
        "judge_model": config.judge_model,
        "n_evaluated": int(len(df)),
        "sample_size_requested": int(config.sample_size),
        "seed": int(config.seed),
        "text_field": config.text_field,
    }

    for field in score_fields:
        summary[f"{field}_mean"] = float(df[field].mean())
        summary[f"{field}_std"] = float(df[field].std(ddof=1)) if len(df) > 1 else 0.0
        summary[f"{field}_median"] = float(df[field].median())

    df["Overall_Mean"] = df[score_fields].mean(axis=1)

    summary["Overall_Mean_mean"] = float(df["Overall_Mean"].mean())
    summary["Overall_Mean_std"] = (
        float(df["Overall_Mean"].std(ddof=1)) if len(df) > 1 else 0.0
    )
    summary["Overall_Mean_median"] = float(df["Overall_Mean"].median())

    return summary


def run_evaluation(
    config: EvalConfig,
    overwrite: bool,
) -> None:
    input_path = Path(config.input_file)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / f"{config.run_name}_judge_results.jsonl"
    results_csv = output_dir / f"{config.run_name}_judge_results.csv"
    summary_json = output_dir / f"{config.run_name}_judge_summary.json"
    config_json = output_dir / f"{config.run_name}_judge_config.json"

    if overwrite:
        for path in (results_jsonl, results_csv, summary_json, config_json):
            if path.exists():
                path.unlink()

    save_json(asdict(config), config_json)

    records = read_input_records(input_path)
    eval_items = prepare_eval_items(
        records=records,
        text_field=config.text_field,
        id_field=config.id_field,
        sample_size=config.sample_size,
        seed=config.seed,
    )

    LOGGER.info("Loaded %d input records.", len(records))
    LOGGER.info("Prepared %d valid evaluation items.", len(eval_items))
    LOGGER.info("Judge model: %s", config.judge_model)
    LOGGER.info("Results JSONL: %s", results_jsonl)

    completed_rows = load_completed_source_rows(results_jsonl)
    pending_items = [
        item for item in eval_items if int(item["source_row"]) not in completed_rows
    ]

    LOGGER.info("Already completed: %d", len(completed_rows))
    LOGGER.info("Pending: %d", len(pending_items))

    client = make_openai_client(
        api_key_env=config.api_key_env,
        base_url=config.base_url,
    )

    for item in tqdm(pending_items, desc="LLM judge evaluation"):
        question = item["question"]

        try:
            judge_result = evaluate_question_via_api(
                client=client,
                question_text=question,
                model=config.judge_model,
                temperature=config.temperature,
                max_retries=config.max_retries,
                retry_sleep_seconds=config.retry_sleep_seconds,
            )
        except Exception as exc:
            LOGGER.error(
                "Failed to evaluate source_row=%s source_id=%s: %s",
                item["source_row"],
                item["source_id"],
                exc,
            )
            continue

        output_row = {
            "source_row": int(item["source_row"]),
            "source_id": item["source_id"],
            "Question": question,
            **judge_result,
        }

        append_jsonl(results_jsonl, [output_row])

        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)

    result_rows = []
    if results_jsonl.exists():
        with results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result_rows.append(json.loads(line))

    df_eval = pd.DataFrame(result_rows)

    if df_eval.empty:
        raise RuntimeError(
            "No evaluation results were produced. Check API configuration and input data."
        )

    df_eval = df_eval.sort_values("source_row").reset_index(drop=True)

    summary = compute_summary(df_eval, config)

    df_eval.to_csv(results_csv, index=False)
    save_json(summary, summary_json)

    print("\n" + "=" * 60)
    print("QUALITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Run name:                  {config.run_name}")
    print(f"Evaluated questions:       {summary['n_evaluated']}")
    print(f"Judge model:               {config.judge_model}")
    print("-" * 60)
    print(f"Average Fluency:           {summary['Fluency_mean']:.2f} / 5.0")
    print(f"Average Clinical Validity: {summary['Clinical_Validity_mean']:.2f} / 5.0")
    print(f"Average Answerability:     {summary['Answerability_mean']:.2f} / 5.0")
    print(f"Average Overall Mean:      {summary['Overall_Mean_mean']:.2f} / 5.0")
    print("=" * 60)
    print(f"Detailed JSONL: {results_jsonl}")
    print(f"Detailed CSV:   {results_csv}")
    print(f"Summary JSON:   {summary_json}")

    if len(df_eval) >= 3:
        print("\nSample evaluations:")
        sample_rows = df_eval.sample(3, random_state=config.seed)
        for _, row in sample_rows.iterrows():
            print(f"\nQ: {row['Question']}")
            print(
                f"   F:{row['Fluency']} | "
                f"V:{row['Clinical_Validity']} | "
                f"A:{row['Answerability']}"
            )
            print(f"   Reason: {row['Reasoning']}")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = make_config(args)
    run_evaluation(config=config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()