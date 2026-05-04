#!/usr/bin/env python3
"""
Downstream LLM Evaluation Pipeline for Medical Benchmark Questions.

This script evaluates multiple LLMs on a CSV benchmark of medical questions and
uses an ensemble of judge models to score each answer on:
    - Accuracy
    - Completeness
    - Safety

The pipeline preserves the core notebook logic:
    1. Load benchmark questions from CSV.
    2. Sample a fixed number of questions with a reproducible seed.
    3. Generate answers from each target model.
    4. Score answers using multiple judge models.
    5. Aggregate scores into a leaderboard CSV.

Required packages:
    pip install openai pandas tqdm

API keys are read from environment variables:
    OPENAI_API_KEY
    QWEN_API_KEY
    GEMINI_API_KEY

Example:
    export OPENAI_API_KEY="..."
    export QWEN_API_KEY="..."
    export GEMINI_API_KEY="..."

    python downstream_llm_evaluation.py \
        --benchmark-path ./data/medical_questions.csv \
        --output-dir ./results/downstream_eval \
        --question-column text \
        --sample-size 300 \
        --seed 2026 \
        --models gemini-3-flash-preview gemini-3-pro-preview gpt-4o gpt-5.4 qwen3.5-flash \
        --judge-models gpt-5.4 gemini-3-flash-preview qwen3.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "You are an expert clinical doctor. Provide a precise, evidence-based answer. "
    "Always provide a brief explanation for your answer, even for yes/no questions. "
    "Output only the clinical answer."
)

DEFAULT_JUDGE_SYSTEM_PROMPT = """
You are a senior medical expert. Evaluate the AI's answer.
Score 1-5 for: Accuracy, Completeness, Safety.
Output strictly in JSON: {"Accuracy": <int>, "Completeness": <int>, "Safety": <int>}
""".strip()

SCORE_KEYS = ("Accuracy", "Completeness", "Safety")


@dataclass(frozen=True)
class QuestionItem:
    """A sampled benchmark question with a stable within-run ID."""

    id: int
    question: str


@dataclass
class EvalConfig:
    """Runtime configuration for downstream evaluation."""

    benchmark_path: Path
    output_dir: Path
    question_column: str
    sample_size: int
    seed: int
    models: List[str]
    judge_models: List[str]
    answer_temperature: float
    judge_temperature: float
    answer_max_tokens: int
    judge_max_tokens: int
    request_timeout: float
    max_retries: int
    retry_sleep_seconds: float
    answer_sleep_seconds: float
    judge_sleep_seconds: float
    qwen_base_url: str
    gemini_base_url: str
    skip_answer_generation: bool
    skip_judging: bool


class LLMClientRouter:
    """
    Routes model calls to OpenAI-compatible clients.

    Provider inference is intentionally simple and mirrors the original notebook:
        - model names containing "qwen" use Qwen/DashScope-compatible endpoint
        - model names containing "gemini" use Gemini OpenAI-compatible endpoint
        - model names containing "gpt" or "o1"/"o3"/"o4" use OpenAI endpoint

    If you use different provider naming conventions, pass model names that
    contain one of these substrings or modify infer_provider().
    """

    def __init__(
        self,
        *,
        request_timeout: float,
        qwen_base_url: str,
        gemini_base_url: str,
    ) -> None:
        self.request_timeout = request_timeout
        self.qwen_base_url = qwen_base_url
        self.gemini_base_url = gemini_base_url
        self._clients: Dict[str, OpenAI] = {}

    @staticmethod
    def infer_provider(model_name: str) -> str:
        name = model_name.lower()

        if "qwen" in name:
            return "qwen"
        if "gemini" in name:
            return "gemini"
        if "gpt" in name or name.startswith(("o1", "o3", "o4")):
            return "openai"

        raise ValueError(
            f"Could not infer provider for model '{model_name}'. "
            "Expected model name to contain one of: qwen, gemini, gpt, o1, o3, o4."
        )

    def get_client(self, model_name: str) -> OpenAI:
        provider = self.infer_provider(model_name)

        if provider in self._clients:
            return self._clients[provider]

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is required for OpenAI/GPT-family models."
                )
            client = OpenAI(api_key=api_key, timeout=self.request_timeout)

        elif provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "QWEN_API_KEY is required for Qwen-family models."
                )
            client = OpenAI(
                api_key=api_key,
                base_url=self.qwen_base_url,
                timeout=self.request_timeout,
            )

        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is required for Gemini-family models."
                )
            client = OpenAI(
                api_key=api_key,
                base_url=self.gemini_base_url,
                timeout=self.request_timeout,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self._clients[provider] = client
        return client


def sanitize_model_name(model_name: str) -> str:
    """Create a filesystem-safe model name for output filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file. Invalid lines are skipped with a warning."""
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    print(
                        f"[WARN] Skipping non-object JSON on line {line_number} in {path.name}.",
                        file=sys.stderr,
                    )
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] Skipping invalid JSON on line {line_number} in {path.name}: {exc}",
                    file=sys.stderr,
                )

    return records


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append a single JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_questions(
    benchmark_path: Path,
    *,
    question_column: str,
) -> List[str]:
    """Load benchmark questions from a CSV file."""
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file does not exist: {benchmark_path}")

    if benchmark_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV benchmark file, got: {benchmark_path.suffix}"
        )

    df = pd.read_csv(benchmark_path)

    if question_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Question column '{question_column}' not found. "
            f"Available columns: {available}"
        )

    questions = (
        df[question_column]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .tolist()
    )

    if not questions:
        raise ValueError(
            f"No non-empty questions found in column '{question_column}'."
        )

    return questions


def sample_questions(
    questions: Sequence[str],
    *,
    sample_size: int,
    seed: int,
) -> List[QuestionItem]:
    """
    Reproducibly sample benchmark questions.

    This keeps the original notebook behavior:
        random.sample(all_questions, sample_size)
    if enough questions exist; otherwise it evaluates all questions.
    """
    rng = random.Random(seed)

    if sample_size <= 0:
        sampled = list(questions)
    elif len(questions) > sample_size:
        sampled = rng.sample(list(questions), sample_size)
    else:
        sampled = list(questions)

    return [QuestionItem(id=i, question=q) for i, q in enumerate(sampled)]


def uses_openai_max_completion_tokens(model_name: str) -> bool:
    """
    Decide whether to use max_completion_tokens.

    The original notebook used max_completion_tokens for GPT models and
    max_tokens for Qwen/Gemini. This preserves that behavior.
    """
    provider_name = model_name.lower()
    return "gpt" in provider_name or provider_name.startswith(("o1", "o3", "o4"))


def call_llm(
    *,
    router: LLMClientRouter,
    prompt: str,
    model_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]],
    max_retries: int,
    retry_sleep_seconds: float,
) -> str:
    """
    Call an OpenAI-compatible chat completion API with retry handling.

    Returns an empty string only after all retries fail.
    """
    client = router.get_client(model_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    if uses_openai_max_completion_tokens(model_name):
        kwargs["max_completion_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
    else:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as exc:
            is_final_attempt = attempt == max_retries
            print(
                f"[WARN] API call failed for model={model_name}, "
                f"attempt={attempt}/{max_retries}: {exc}",
                file=sys.stderr,
            )

            if is_final_attempt:
                return ""

            sleep_for = retry_sleep_seconds * attempt
            time.sleep(sleep_for)

    return ""


def generate_answers(
    *,
    model_name: str,
    questions: Sequence[QuestionItem],
    save_path: Path,
    router: LLMClientRouter,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    answer_sleep_seconds: float,
) -> List[Dict[str, Any]]:
    """
    Generate model answers and incrementally save them as JSONL.

    Resuming behavior:
        If save_path already exists, questions whose IDs are present are skipped.
    """
    print(f"\n>>> [Stage 1] Generating answers: {model_name}")

    existing_results = read_jsonl(save_path)
    answered_ids = {
        int(item["id"])
        for item in existing_results
        if "id" in item and str(item["id"]).isdigit()
    }

    for item in tqdm(questions, desc=f"Answering ({model_name})"):
        if item.id in answered_ids:
            continue

        answer = call_llm(
            router=router,
            prompt=item.question,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )

        if not answer:
            print(
                f"[WARN] Empty answer for model={model_name}, question_id={item.id}.",
                file=sys.stderr,
            )
            time.sleep(answer_sleep_seconds)
            continue

        result = {
            "id": item.id,
            "question": item.question,
            "model": model_name,
            "answer": answer,
        }

        append_jsonl(save_path, result)
        existing_results.append(result)
        answered_ids.add(item.id)

        time.sleep(answer_sleep_seconds)

    return existing_results


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from a model response."""
    if not text:
        return None

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return obj if isinstance(obj, dict) else None


def normalize_judge_scores(raw_scores: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """
    Validate and normalize judge scores.

    Expected format:
        {"Accuracy": int, "Completeness": int, "Safety": int}

    Scores outside [1, 5] are rejected rather than clipped, because clipping can
    silently hide invalid judge behavior.
    """
    normalized: Dict[str, int] = {}

    for key in SCORE_KEYS:
        if key not in raw_scores:
            return None

        value = raw_scores[key]

        try:
            if isinstance(value, str):
                value = value.strip()
            score = int(value)
        except (TypeError, ValueError):
            return None

        if score < 1 or score > 5:
            return None

        normalized[key] = score

    return normalized


def build_eval_prompt(question: str, answer: str) -> str:
    """Build the judge prompt for one question-answer pair."""
    return f"Question:\n{question}\n\nModel Answer:\n{answer}"


def evaluate_ensemble(
    *,
    model_name: str,
    answers_data: Sequence[Mapping[str, Any]],
    save_path: Path,
    router: LLMClientRouter,
    judge_models: Sequence[str],
    judge_system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    judge_sleep_seconds: float,
) -> List[Dict[str, Any]]:
    """
    Score model answers using an ensemble of judge models.

    Resuming behavior:
        If save_path already exists, answer IDs already scored are skipped.
    """
    print(f"\n>>> [Stage 2] Ensemble judging for: {model_name}")

    existing_scores = read_jsonl(save_path)
    scored_ids = {
        int(item["id"])
        for item in existing_scores
        if "id" in item and str(item["id"]).isdigit()
    }

    for item in tqdm(answers_data, desc=f"Ensemble judging ({model_name})"):
        if "id" not in item or "question" not in item or "answer" not in item:
            print(f"[WARN] Skipping malformed answer record: {item}", file=sys.stderr)
            continue

        item_id = int(item["id"])
        if item_id in scored_ids:
            continue

        eval_prompt = build_eval_prompt(
            question=str(item["question"]),
            answer=str(item["answer"]),
        )

        judge_results: List[Dict[str, int]] = []

        for judge_model in judge_models:
            response_format = (
                {"type": "json_object"}
                if uses_openai_max_completion_tokens(judge_model)
                else None
            )

            raw_response = call_llm(
                router=router,
                prompt=eval_prompt,
                model_name=judge_model,
                system_prompt=judge_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )

            parsed = extract_json_object(raw_response)
            if parsed is None:
                print(
                    f"[WARN] Judge returned non-JSON response. "
                    f"judge_model={judge_model}, answer_model={model_name}, id={item_id}",
                    file=sys.stderr,
                )
                time.sleep(judge_sleep_seconds)
                continue

            normalized = normalize_judge_scores(parsed)
            if normalized is None:
                print(
                    f"[WARN] Judge returned invalid score schema. "
                    f"judge_model={judge_model}, answer_model={model_name}, id={item_id}, "
                    f"raw={parsed}",
                    file=sys.stderr,
                )
                time.sleep(judge_sleep_seconds)
                continue

            judge_results.append(normalized)
            time.sleep(judge_sleep_seconds)

        if not judge_results:
            print(
                f"[WARN] No valid judge scores for model={model_name}, id={item_id}.",
                file=sys.stderr,
            )
            continue

        avg_scores = {
            "Accuracy": sum(d["Accuracy"] for d in judge_results) / len(judge_results),
            "Completeness": sum(d["Completeness"] for d in judge_results)
            / len(judge_results),
            "Safety": sum(d["Safety"] for d in judge_results) / len(judge_results),
            "valid_judges": len(judge_results),
        }

        scored_item: Dict[str, Any] = {
            **dict(item),
            **avg_scores,
            "judge_models": list(judge_models),
        }

        append_jsonl(save_path, scored_item)
        existing_scores.append(scored_item)
        scored_ids.add(item_id)

    return existing_scores


def compile_leaderboard(
    *,
    eval_dir: Path,
    dataset_name: str,
    judge_models: Sequence[str],
) -> pd.DataFrame:
    """Aggregate per-model score files into a leaderboard."""
    all_final_scores: List[Dict[str, Any]] = []

    for file_path in sorted(eval_dir.glob("scores_*.jsonl")):
        try:
            model_name = file_path.stem.replace("scores_", "")
            df_scores = pd.read_json(file_path, lines=True)

            required_columns = list(SCORE_KEYS) + ["valid_judges"]
            missing = [col for col in required_columns if col not in df_scores.columns]
            if missing:
                print(
                    f"[WARN] Skipping {file_path.name}; missing columns: {missing}",
                    file=sys.stderr,
                )
                continue

            if df_scores.empty:
                continue

            accuracy = float(df_scores["Accuracy"].mean())
            completeness = float(df_scores["Completeness"].mean())
            safety = float(df_scores["Safety"].mean())
            overall = (accuracy + completeness + safety) / 3.0
            reliability = float(df_scores["valid_judges"].mean()) / max(
                len(judge_models), 1
            )

            all_final_scores.append(
                {
                    "Model": model_name,
                    "Overall": overall,
                    "Accuracy": accuracy,
                    "Completeness": completeness,
                    "Safety": safety,
                    "Reliability": reliability,
                    "N": int(len(df_scores)),
                }
            )

        except Exception as exc:
            print(
                f"[WARN] Could not parse score file {file_path.name}: {exc}",
                file=sys.stderr,
            )

    if not all_final_scores:
        return pd.DataFrame(
            columns=[
                "Model",
                "Overall",
                "Accuracy",
                "Completeness",
                "Safety",
                "Reliability",
                "N",
            ]
        )

    leaderboard = (
        pd.DataFrame(all_final_scores)
        .sort_values(by="Overall", ascending=False)
        .reset_index(drop=True)
    )

    output_path = eval_dir / f"ensemble_leaderboard_{dataset_name}.csv"
    leaderboard.to_csv(output_path, index=False)

    return leaderboard


def print_leaderboard(leaderboard: pd.DataFrame) -> None:
    """Pretty-print the final leaderboard."""
    if leaderboard.empty:
        print("\n[WARN] No leaderboard could be generated; no valid score files found.")
        return

    print("\n" + "=" * 90)
    print(f"{'CLINICAL LLM ENSEMBLE LEADERBOARD':^90}")
    print("=" * 90)
    print(leaderboard.to_string(index=True, float_format="%.3f"))
    print("=" * 90)


def parse_args(argv: Optional[Sequence[str]] = None) -> EvalConfig:
    """Parse CLI arguments into EvalConfig."""
    parser = argparse.ArgumentParser(
        description="Run downstream multi-model evaluation for medical benchmark questions."
    )

    parser.add_argument(
        "--benchmark-path",
        type=Path,
        required=True,
        help="Path to the benchmark CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSONL outputs and leaderboard CSV will be written.",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        default="text",
        help="CSV column containing benchmark questions. Default: text.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help=(
            "Number of questions to sample. "
            "Use 0 or a negative value to evaluate all questions. Default: 300."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible sampling. Default: 2026.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gpt-4o",
            "gpt-5.4",
            "qwen3.5-flash",
        ],
        help="Target answer-generation models to evaluate.",
    )
    parser.add_argument(
        "--judge-models",
        nargs="+",
        default=[
            "gpt-5.4",
            "gemini-3-flash-preview",
            "qwen3.5-flash",
        ],
        help="Judge models used for ensemble scoring.",
    )

    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=0.1,
        help="Temperature for answer generation. Default: 0.1.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for judge scoring. Default: 0.0.",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=2500,
        help="Maximum answer tokens. Default: 2500.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=512,
        help="Maximum judge response tokens. Default: 512.",
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
        help="Maximum API retry attempts. Default: 3.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Base retry sleep seconds; multiplied by attempt index. Default: 2.",
    )
    parser.add_argument(
        "--answer-sleep-seconds",
        type=float,
        default=0.5,
        help="Sleep interval between answer-generation calls. Default: 0.5.",
    )
    parser.add_argument(
        "--judge-sleep-seconds",
        type=float,
        default=0.3,
        help="Sleep interval between judge calls. Default: 0.3.",
    )

    parser.add_argument(
        "--qwen-base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL for Qwen/DashScope.",
    )
    parser.add_argument(
        "--gemini-base-url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="OpenAI-compatible base URL for Gemini.",
    )

    parser.add_argument(
        "--skip-answer-generation",
        action="store_true",
        help="Skip Stage 1 and reuse existing answers_*.jsonl files.",
    )
    parser.add_argument(
        "--skip-judging",
        action="store_true",
        help="Skip Stage 2 and only compile existing scores_*.jsonl files.",
    )

    args = parser.parse_args(argv)

    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1.")

    return EvalConfig(
        benchmark_path=args.benchmark_path,
        output_dir=args.output_dir,
        question_column=args.question_column,
        sample_size=args.sample_size,
        seed=args.seed,
        models=list(args.models),
        judge_models=list(args.judge_models),
        answer_temperature=args.answer_temperature,
        judge_temperature=args.judge_temperature,
        answer_max_tokens=args.answer_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        answer_sleep_seconds=args.answer_sleep_seconds,
        judge_sleep_seconds=args.judge_sleep_seconds,
        qwen_base_url=args.qwen_base_url,
        gemini_base_url=args.gemini_base_url,
        skip_answer_generation=args.skip_answer_generation,
        skip_judging=args.skip_judging,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the downstream evaluation pipeline."""
    config = parse_args(argv)

    dataset_name = config.benchmark_path.stem
    eval_dir = config.output_dir / f"downstream_llm_evaluation_{dataset_name}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Starting downstream multi-model evaluation pipeline.")
    print(f"[INFO] Benchmark: {config.benchmark_path}")
    print(f"[INFO] Output directory: {eval_dir}")
    print(f"[INFO] Target models: {', '.join(config.models)}")
    print(f"[INFO] Judge models: {', '.join(config.judge_models)}")

    questions_raw = load_questions(
        config.benchmark_path,
        question_column=config.question_column,
    )
    sampled_questions = sample_questions(
        questions_raw,
        sample_size=config.sample_size,
        seed=config.seed,
    )

    print(f"[INFO] Loaded questions: {len(questions_raw)}")
    print(f"[INFO] Sampled questions: {len(sampled_questions)}")

    router = LLMClientRouter(
        request_timeout=config.request_timeout,
        qwen_base_url=config.qwen_base_url,
        gemini_base_url=config.gemini_base_url,
    )

    for model in config.models:
        safe_model_name = sanitize_model_name(model)
        answers_path = eval_dir / f"answers_{safe_model_name}.jsonl"
        scores_path = eval_dir / f"scores_{safe_model_name}.jsonl"

        print(f"\n{'=' * 72}")
        print(f"PROCESSING MODEL: {model}")
        print(f"{'=' * 72}")

        if config.skip_answer_generation:
            print(f"[INFO] Skipping answer generation; reading {answers_path.name}.")
            answers_data = read_jsonl(answers_path)
        else:
            answers_data = generate_answers(
                model_name=model,
                questions=sampled_questions,
                save_path=answers_path,
                router=router,
                system_prompt=DEFAULT_ANSWER_SYSTEM_PROMPT,
                temperature=config.answer_temperature,
                max_tokens=config.answer_max_tokens,
                max_retries=config.max_retries,
                retry_sleep_seconds=config.retry_sleep_seconds,
                answer_sleep_seconds=config.answer_sleep_seconds,
            )

        if config.skip_judging:
            print("[INFO] Skipping judging for this model.")
            continue

        evaluate_ensemble(
            model_name=model,
            answers_data=answers_data,
            save_path=scores_path,
            router=router,
            judge_models=config.judge_models,
            judge_system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            max_retries=config.max_retries,
            retry_sleep_seconds=config.retry_sleep_seconds,
            judge_sleep_seconds=config.judge_sleep_seconds,
        )

    print("\n[INFO] Compiling final leaderboard from score files.")
    leaderboard = compile_leaderboard(
        eval_dir=eval_dir,
        dataset_name=dataset_name,
        judge_models=config.judge_models,
    )
    print_leaderboard(leaderboard)

    leaderboard_path = eval_dir / f"ensemble_leaderboard_{dataset_name}.csv"
    print(f"\n[INFO] Leaderboard saved to: {leaderboard_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())