#!/usr/bin/env python3
"""
Cross-model aggregation for LLM-judge evaluation results.

This script summarizes evaluation CSV files produced by multiple judge models
for the full pipeline and ablation studies.

Aggregation method:
  1. For each experiment and each judge model, compute the mean of each score
     column within that judge model's CSV.
  2. For each experiment, average those per-judge-model means across judge
     models with equal weight.

This preserves the original intended method:
  - equal weight per judge model
  - not pooled row-level averaging across all judge outputs

Experiments covered by default:
  - Full Pipeline
  - No Question Classifier
  - No Gibberish Detector
  - No LLM-Rewrite

The "No LLM-Rewrite" experiment corresponds to evaluating pre-rewrite outputs,
for example Step 3 clean texts, to isolate the contribution of the LLM rewriting
stage.

Expected default directory layout:
  evaluation-root/
    gpt-4o-mini/
      full_true_evaluation.csv
      nofate_noquestion_evaluation.csv
      no_gibberish_evaluation_gpt-5.4.csv
      nollm_evaluation.csv
    gemini3_flash/
      ...
    qwen3.5_flash/
      ...

Example:
  python summarize_cross_model_evaluation.py \\
    --evaluation-root ./new_generation_results \\
    --output-dir ./new_generation_results

Example with custom experiment files:
  python summarize_cross_model_evaluation.py \\
    --evaluation-root ./eval \\
    --model-folders gpt-4o-mini gemini3_flash qwen3.5_flash \\
    --experiment-file "Full Pipeline=full_pipeline_evaluation.csv" \\
    --experiment-file "No LLM-Rewrite=no_llm_rewrite_evaluation.csv"

Outputs:
  output-dir/
    cross_model_final_summary.csv
    cross_model_model_level_means.csv
    cross_model_missing_or_invalid_files.csv
    cross_model_final_summary.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_FOLDERS: List[str] = [
    "gpt-4o-mini",
    "gemini3_flash",
    "qwen3.5_flash",
]


# Legacy filenames are preserved for compatibility with existing results.
DEFAULT_EXPERIMENT_FILES: Dict[str, str] = {
    "Full Pipeline": "full_true_evaluation.csv",
    "No Question Classifier": "nofate_noquestion_evaluation.csv",
    "No Gibberish Detector": "no_gibberish_evaluation_gpt-5.4.csv",
    "No LLM-Rewrite": "nollm_evaluation.csv",
}


DEFAULT_SCORE_COLUMNS: List[str] = [
    "Fluency",
    "Clinical_Validity",
    "Clinical_Depth_Difficulty",
    "Answerability",
]


@dataclass(frozen=True)
class AggregationConfig:
    evaluation_root: str
    output_dir: str
    model_folders: List[str]
    experiment_files: Dict[str, str]
    score_columns: List[str]
    require_all_score_columns: bool
    output_prefix: str


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_experiment_file_arg(value: str) -> Tuple[str, str]:
    """
    Parse CLI argument of the form:
      Experiment Name=filename.csv
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --experiment-file value: {value!r}. "
            "Expected format: 'Experiment Name=filename.csv'."
        )

    name, filename = value.split("=", 1)
    name = name.strip()
    filename = filename.strip()

    if not name or not filename:
        raise argparse.ArgumentTypeError(
            f"Invalid --experiment-file value: {value!r}. "
            "Both experiment name and filename must be non-empty."
        )

    return name, filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation CSVs across multiple LLM judge models."
    )

    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
        help="Root directory containing one subdirectory per judge model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where summary files will be saved. Default: evaluation-root.",
    )
    parser.add_argument(
        "--model-folders",
        nargs="+",
        default=DEFAULT_MODEL_FOLDERS,
        help="Judge-model subdirectories under evaluation-root.",
    )
    parser.add_argument(
        "--experiment-file",
        action="append",
        type=parse_experiment_file_arg,
        default=None,
        help=(
            "Experiment-to-filename mapping. Can be repeated. "
            "Format: 'Experiment Name=filename.csv'. "
            "If omitted, default mappings are used."
        ),
    )
    parser.add_argument(
        "--score-columns",
        nargs="+",
        default=DEFAULT_SCORE_COLUMNS,
        help="Score columns to aggregate if present.",
    )
    parser.add_argument(
        "--require-all-score-columns",
        action="store_true",
        help=(
            "If set, skip a file unless it contains all requested score columns. "
            "By default, files are allowed to contain any subset of score columns."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="cross_model",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def make_config(args: argparse.Namespace) -> AggregationConfig:
    evaluation_root = args.evaluation_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_root
    )

    if args.experiment_file:
        experiment_files = dict(args.experiment_file)
    else:
        experiment_files = dict(DEFAULT_EXPERIMENT_FILES)

    return AggregationConfig(
        evaluation_root=str(evaluation_root),
        output_dir=str(output_dir),
        model_folders=list(args.model_folders),
        experiment_files=experiment_files,
        score_columns=list(args.score_columns),
        require_all_score_columns=bool(args.require_all_score_columns),
        output_prefix=args.output_prefix,
    )


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file {path}: {exc}") from exc


def numeric_mean(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def summarize_one_file(
    csv_path: Path,
    score_columns: Sequence[str],
    require_all_score_columns: bool,
) -> Tuple[Optional[Dict[str, float]], Dict[str, object]]:
    """
    Compute metric means for one judge-model CSV.

    Returns:
      metric_means:
        Dict of metric -> mean, or None if the file cannot be used.
      status:
        Diagnostic metadata for missing/invalid reporting.
    """
    status: Dict[str, object] = {
        "file_path": str(csv_path),
        "exists": csv_path.exists(),
        "usable": False,
        "reason": "",
        "rows": 0,
        "available_score_columns": "",
        "missing_score_columns": "",
    }

    if not csv_path.exists():
        status["reason"] = "missing_file"
        return None, status

    try:
        df = safe_read_csv(csv_path)
    except Exception as exc:
        status["reason"] = f"read_error: {exc}"
        return None, status

    status["rows"] = int(len(df))

    existing_cols = [col for col in score_columns if col in df.columns]
    missing_cols = [col for col in score_columns if col not in df.columns]

    status["available_score_columns"] = ",".join(existing_cols)
    status["missing_score_columns"] = ",".join(missing_cols)

    if require_all_score_columns and missing_cols:
        status["reason"] = "missing_required_score_columns"
        return None, status

    if not existing_cols:
        status["reason"] = "no_score_columns_found"
        return None, status

    metric_means: Dict[str, float] = {}

    for col in existing_cols:
        mean_value = numeric_mean(df[col])
        if mean_value is not None:
            metric_means[col] = mean_value

    if not metric_means:
        status["reason"] = "score_columns_non_numeric_or_empty"
        return None, status

    status["usable"] = True
    status["reason"] = "ok"

    return metric_means, status


def aggregate_cross_model(config: AggregationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-experiment scores across judge models.

    Returns:
      final_summary_df:
        One row per experiment, with cross-model metric means.
      model_level_df:
        One row per experiment x judge model, with that judge model's means.
      diagnostics_df:
        One row per expected CSV file, including missing/invalid status.
    """
    evaluation_root = Path(config.evaluation_root)

    final_rows: List[Dict[str, object]] = []
    model_level_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    LOGGER.info("Starting cross-model aggregation.")
    LOGGER.info("Evaluation root: %s", evaluation_root)

    for experiment_name, filename in config.experiment_files.items():
        LOGGER.info("Processing experiment: %s", experiment_name)

        experiment_model_means: List[Dict[str, float]] = []

        for model_folder in config.model_folders:
            csv_path = evaluation_root / model_folder / filename

            metric_means, status = summarize_one_file(
                csv_path=csv_path,
                score_columns=config.score_columns,
                require_all_score_columns=config.require_all_score_columns,
            )

            diagnostic_row = {
                "Experiment": experiment_name,
                "Judge_Model": model_folder,
                "Expected_File": filename,
                **status,
            }
            diagnostic_rows.append(diagnostic_row)

            if metric_means is None:
                LOGGER.warning(
                    "Skipping %s / %s: %s",
                    experiment_name,
                    model_folder,
                    status["reason"],
                )
                continue

            experiment_model_means.append(metric_means)

            model_row: Dict[str, object] = {
                "Experiment": experiment_name,
                "Judge_Model": model_folder,
                "File": str(csv_path),
                "Rows": int(status["rows"]),
            }
            for metric in config.score_columns:
                model_row[metric] = metric_means.get(metric)

            model_level_rows.append(model_row)

            LOGGER.info("  Loaded scores from judge model: %s", model_folder)

        if not experiment_model_means:
            LOGGER.error("No usable score data for experiment: %s", experiment_name)
            continue

        final_row: Dict[str, object] = {
            "Experiment": experiment_name,
            "Models_Count": len(experiment_model_means),
        }

        for metric in config.score_columns:
            metric_values = [
                model_means[metric]
                for model_means in experiment_model_means
                if metric in model_means
            ]

            if metric_values:
                final_row[metric] = float(pd.Series(metric_values).mean())
                final_row[f"{metric}_Judge_Count"] = int(len(metric_values))
            else:
                final_row[metric] = None
                final_row[f"{metric}_Judge_Count"] = 0

        available_metric_cols = [
            metric for metric in config.score_columns if final_row.get(metric) is not None
        ]
        if available_metric_cols:
            final_row["Overall_Mean"] = float(
                pd.Series([final_row[metric] for metric in available_metric_cols]).mean()
            )
        else:
            final_row["Overall_Mean"] = None

        final_rows.append(final_row)

    final_summary_df = pd.DataFrame(final_rows)
    model_level_df = pd.DataFrame(model_level_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)

    return final_summary_df, model_level_df, diagnostics_df


def order_summary_columns(
    df: pd.DataFrame,
    score_columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    preferred = ["Experiment", "Models_Count"]
    for metric in score_columns:
        preferred.append(metric)
        preferred.append(f"{metric}_Judge_Count")
    preferred.append("Overall_Mean")

    existing_preferred = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_preferred]

    return df[existing_preferred + remaining]


def save_outputs(
    final_summary_df: pd.DataFrame,
    model_level_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    config: AggregationConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / f"{config.output_prefix}_final_summary.csv"
    model_level_csv = output_dir / f"{config.output_prefix}_model_level_means.csv"
    diagnostics_csv = output_dir / f"{config.output_prefix}_missing_or_invalid_files.csv"
    summary_md = output_dir / f"{config.output_prefix}_final_summary.md"
    config_json = output_dir / f"{config.output_prefix}_aggregation_config.json"

    final_summary_df = order_summary_columns(final_summary_df, config.score_columns)

    final_summary_df.to_csv(summary_csv, index=False)
    model_level_df.to_csv(model_level_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    try:
        final_summary_df.to_markdown(summary_md, index=False)
    except Exception:
        summary_md.write_text(
            final_summary_df.to_string(index=False),
            encoding="utf-8",
        )

    with config_json.open("w", encoding="utf-8") as f:
        import json

        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    LOGGER.info("Saved final summary: %s", summary_csv)
    LOGGER.info("Saved model-level means: %s", model_level_csv)
    LOGGER.info("Saved diagnostics: %s", diagnostics_csv)
    LOGGER.info("Saved markdown summary: %s", summary_md)
    LOGGER.info("Saved aggregation config: %s", config_json)


def print_summary(final_summary_df: pd.DataFrame, config: AggregationConfig) -> None:
    if final_summary_df.empty:
        print("\n[ERROR] No data could be aggregated. Check paths and filenames.")
        return

    display_df = order_summary_columns(final_summary_df, config.score_columns)

    print("\n" + "=" * 88)
    print("Cross-Model Evaluation Summary")
    print("Averaged across judge models with equal model weight")
    print("=" * 88)
    print(display_df.to_string(index=False))
    print("=" * 88)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config = make_config(args)

    final_summary_df, model_level_df, diagnostics_df = aggregate_cross_model(config)

    save_outputs(
        final_summary_df=final_summary_df,
        model_level_df=model_level_df,
        diagnostics_df=diagnostics_df,
        config=config,
    )

    print_summary(final_summary_df, config)


if __name__ == "__main__":
    main()