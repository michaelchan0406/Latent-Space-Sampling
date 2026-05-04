#!/usr/bin/env python3
"""
make_tables_and_figures.py

Generate reviewer-facing tables and figures by combining:
  - distribution similarity cache: MMD, FBD, KL
  - diversity cache: Dist-1, Self-BLEU, Dist-2, V-EE, Topic Coverage,
    Topic Entropy, Vendi Score, Gzip Ratio, Mean Pairwise Distance

Outputs:
  - raw combined metrics CSV
  - normalized metrics CSV
  - LaTeX table
  - radar figure PNG/PDF

The radar normalization follows the notebook logic:
  - lower-is-better metrics use min(value) / value
  - higher-is-better metrics use value / max(value)
  - normalization is performed within each model family
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------

SIM_METRICS = ["MMD", "FBD", "KL"]

DIV_METRICS = [
    "Dist-1",
    "Self-BLEU",
    "Dist-2",
    "V-EE",
    "Topic Cov (%)",
    "Topic Ent",
    "Vendi Score",
    "Gzip Ratio",
    "Mean Pairwise Dist",
]

ALL_METRICS = SIM_METRICS + DIV_METRICS

LOWER_IS_BETTER = {
    "MMD",
    "FBD",
    "KL",
    "Self-BLEU",
}

METRIC_LABELS = {
    "MMD": "MMD ↓",
    "FBD": "FBD ↓",
    "KL": "KL ↓",
    "Dist-1": "Dist-1 ↑",
    "Self-BLEU": "Self-BLEU ↓",
    "Dist-2": "Dist-2 ↑",
    "V-EE": "V-EE ↑",
    "Topic Cov (%)": "Topic Cov ↑",
    "Topic Ent": "Topic Ent ↑",
    "Vendi Score": "Vendi ↑",
    "Gzip Ratio": "Gzip ↑",
    "Mean Pairwise Dist": "Pairwise Dist ↑",
}

METHOD_ORDER = [
    "M1",
    "M2",
    "Med-RAG",
    "Self-Instruct",
    "WizardLM",
    "Stratified Sampling",
    "Global Sampling",
]

METHOD_COLORS = {
    "M1": "#1f77b4",
    "M2": "#ff7f0e",
    "Med-RAG": "#2ca02c",
    "Self-Instruct": "#e377c2",
    "WizardLM": "#9467bd",
    "Global Sampling": "#d62728",
    "Stratified Sampling": "#000000",
}

# Default compatibility configuration for names used in earlier notebooks.
# You can override this with --family_config.
DEFAULT_FAMILY_CONFIG = {
    "GPT-5.4": [
        ("M1", ["M1 GPT-5.4", "m1__gpt-5.4"]),
        ("M2", ["M2 GPT-5.4", "M4 GPT-5.4", "m2__gpt-5.4"]),
        ("Med-RAG", ["Med-Rag-gpt-5.4", "Med-RAG-gpt-5.4", "med_rag__gpt-5.4"]),
        ("Self-Instruct", ["Self-instruct_gpt-5.4", "self_instruct__gpt-5.4"]),
        ("WizardLM", ["WizardLM-gpt-5.4", "wizardlm__gpt-5.4"]),
        ("Stratified Sampling", ["Ours-gpt-5.4-stratifiedL2"]),
        ("Global Sampling", ["global_gmm_gpt5.4"]),
    ],
    "Gemini-3-Flash": [
        ("M1", ["M1 Gemini3-f", "m1__gemini-3-flash-preview"]),
        ("M2", ["M2 Gemini3-f", "M4 Gemini3-f", "m2__gemini-3-flash-preview"]),
        ("Med-RAG", ["Med-Rag-gemini3-f", "Med-RAG-gemini3-f", "med_rag__gemini-3-flash-preview"]),
        ("Self-Instruct", ["Self-instruct_gemini3-f", "self_instruct__gemini-3-flash-preview"]),
        ("WizardLM", ["WizardLM-gemini3-f", "wizardlm__gemini-3-flash-preview"]),
        ("Stratified Sampling", ["Ours-gemini3f-stratifiedL2"]),
        ("Global Sampling", ["global_gmm_gemini3-f"]),
    ],
    "Qwen-3.5": [
        ("M1", ["M1 Qwen3.5-f", "m1__qwen3.5-flash"]),
        ("M2", ["M2 Qwen3.5-f", "M4 Qwen3.5-f", "m2__qwen3.5-flash"]),
        ("Med-RAG", ["Med-Rag-qwen3.5-f", "Med-Rag_qwen3.5-f", "med_rag__qwen3.5-flash"]),
        ("Self-Instruct", ["Self-instruct_qwen3.5-f", "self_instruct__qwen3.5-flash"]),
        ("WizardLM", ["WizardLM-qwen3.5-f", "wizardlm__qwen3.5-flash"]),
        ("Stratified Sampling", ["Ours-qwen3.5f-stratifiedL2"]),
        ("Global Sampling", ["global_gmm_qwen3.5-f"]),
    ],
}


# ---------------------------------------------------------------------
# Logging and cache utilities
# ---------------------------------------------------------------------

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_family_config(path: Optional[str]) -> Dict[str, List[Tuple[str, List[str]]]]:
    if path is None:
        return DEFAULT_FAMILY_CONFIG

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Family config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: Dict[str, List[Tuple[str, List[str]]]] = {}
    for family, items in raw.items():
        parsed_items: List[Tuple[str, List[str]]] = []
        for item in items:
            if isinstance(item, dict):
                display_name = item["display_name"]
                candidates = item["candidate_keys"]
            else:
                display_name, candidates = item
            parsed_items.append((str(display_name), [str(x) for x in candidates]))
        parsed[str(family)] = parsed_items

    return parsed


def find_first_existing_key(candidate_keys: Sequence[str], cache: Dict[str, Any]) -> Optional[str]:
    for key in candidate_keys:
        if key in cache:
            return key
    return None


def extract_metric_mean(
    cache: Dict[str, Any],
    key: str,
    metric: str,
) -> Optional[float]:
    entry = cache.get(key, {})
    plot_data = entry.get("plot_data", {})
    metric_data = plot_data.get(metric)

    if isinstance(metric_data, dict) and metric_data.get("mean") is not None:
        return float(metric_data["mean"])

    if metric in plot_data and isinstance(plot_data[metric], (int, float)):
        return float(plot_data[metric])

    return None


# ---------------------------------------------------------------------
# Dataframe assembly
# ---------------------------------------------------------------------

def assemble_raw_dataframe(
    similarity_cache: Dict[str, Any],
    diversity_cache: Dict[str, Any],
    family_config: Dict[str, List[Tuple[str, List[str]]]],
    require_all_metrics: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for family_name, methods in family_config.items():
        for display_name, candidate_keys in methods:
            sim_key = find_first_existing_key(candidate_keys, similarity_cache)
            div_key = find_first_existing_key(candidate_keys, diversity_cache)

            if sim_key is None or div_key is None:
                logging.warning(
                    "Skipping family=%s method=%s: sim_key=%s div_key=%s",
                    family_name,
                    display_name,
                    sim_key,
                    div_key,
                )
                continue

            row: Dict[str, Any] = {
                "Family": family_name,
                "Display Name": display_name,
                "Similarity Cache Key": sim_key,
                "Diversity Cache Key": div_key,
                "Method Order": (
                    METHOD_ORDER.index(display_name)
                    if display_name in METHOD_ORDER
                    else len(METHOD_ORDER)
                ),
            }

            missing: List[str] = []

            for metric in SIM_METRICS:
                value = extract_metric_mean(similarity_cache, sim_key, metric)
                if value is None:
                    missing.append(metric)
                row[metric] = value

            for metric in DIV_METRICS:
                value = extract_metric_mean(diversity_cache, div_key, metric)
                if value is None:
                    missing.append(metric)
                row[metric] = value

            if missing:
                message = (
                    f"family={family_name}, method={display_name} missing metrics: {missing}"
                )
                if require_all_metrics:
                    logging.warning("Skipping %s", message)
                    continue
                logging.warning("Keeping incomplete row: %s", message)

            rows.append(row)

    if not rows:
        raise RuntimeError("No complete rows found. Check cache keys and family config.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["Family", "Method Order", "Display Name"]).reset_index(drop=True)
    return df


def normalize_within_family(df: pd.DataFrame) -> pd.DataFrame:
    norm_df = df.copy()

    for family_name in norm_df["Family"].unique():
        mask = norm_df["Family"] == family_name

        for metric in ALL_METRICS:
            vals = norm_df.loc[mask, metric].astype(float)

            if metric in LOWER_IS_BETTER:
                min_val = vals.min(skipna=True)
                norm_df.loc[mask, f"{metric}_norm"] = min_val / vals.clip(lower=1e-12)
            else:
                max_val = vals.max(skipna=True)
                norm_df.loc[mask, f"{metric}_norm"] = vals / max(max_val, 1e-12)

    return norm_df


def make_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    display_cols = ["Family", "Display Name"] + ALL_METRICS
    table_df = df[display_cols].copy()

    rename_map = {
        metric: METRIC_LABELS.get(metric, metric)
        for metric in ALL_METRICS
    }
    table_df = table_df.rename(columns=rename_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table_df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4g}",
        escape=True,
    )
    output_path.write_text(latex, encoding="utf-8")


# ---------------------------------------------------------------------
# Radar plotting
# ---------------------------------------------------------------------

def plot_radar(
    norm_df: pd.DataFrame,
    family_config: Dict[str, List[Tuple[str, List[str]]]],
    output_png: Path,
    output_pdf: Optional[Path],
    title: Optional[str],
    include_legend: bool,
) -> None:
    families = list(family_config.keys())
    n_families = len(families)

    num_vars = len(ALL_METRICS)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig_width = max(6.0 * n_families, 10.0)
    fig, axes = plt.subplots(
        1,
        n_families,
        figsize=(fig_width, 6),
        subplot_kw=dict(polar=True),
    )

    if n_families == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    legend_added = set()

    for ax, family_name in zip(axes, families):
        df_sub = norm_df[norm_df["Family"] == family_name].sort_values("Method Order")
        if df_sub.empty:
            ax.set_axis_off()
            continue

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        for _, row in df_sub.iterrows():
            display_name = str(row["Display Name"])
            values = [float(row[f"{metric}_norm"]) for metric in ALL_METRICS]
            values += values[:1]

            color = METHOD_COLORS.get(display_name, "#333333")
            line, = ax.plot(
                angles,
                values,
                linewidth=2,
                label=display_name,
                color=color,
                marker="o",
                markersize=3.5,
            )
            ax.fill(angles, values, alpha=0.08, color=color)

            if display_name not in legend_added:
                legend_handles.append(line)
                legend_labels.append(display_name)
                legend_added.add(display_name)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [METRIC_LABELS[metric] for metric in ALL_METRICS],
            fontsize=8.5,
            fontweight="bold",
        )

        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="gray")
        ax.set_title(family_name, fontsize=16, fontweight="bold", pad=20)

    if title:
        fig.suptitle(title, fontsize=17, fontweight="bold", y=1.02)

    if include_legend and legend_handles:
        ncol = min(4, len(legend_handles))
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=ncol,
            frameon=False,
            fontsize=12,
            handlelength=2.2,
            columnspacing=1.6,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")

    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_pdf, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def generate_tables_and_figures(args: argparse.Namespace) -> None:
    similarity_cache = load_json(Path(args.similarity_cache))
    diversity_cache = load_json(Path(args.diversity_cache))
    family_config = load_family_config(args.family_config)

    raw_df = assemble_raw_dataframe(
        similarity_cache=similarity_cache,
        diversity_cache=diversity_cache,
        family_config=family_config,
        require_all_metrics=not args.allow_incomplete,
    )

    norm_df = normalize_within_family(raw_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = output_dir / args.raw_csv_name
    norm_csv = output_dir / args.normalized_csv_name
    latex_path = output_dir / args.latex_name
    radar_png = output_dir / args.radar_png_name
    radar_pdf = output_dir / args.radar_pdf_name if args.save_pdf else None

    display_cols = [
        "Family",
        "Display Name",
        "Similarity Cache Key",
        "Diversity Cache Key",
    ] + ALL_METRICS

    raw_df[display_cols].to_csv(raw_csv, index=False)

    norm_cols = [
        "Family",
        "Display Name",
    ] + [f"{metric}_norm" for metric in ALL_METRICS]

    norm_df[norm_cols].to_csv(norm_csv, index=False)
    make_latex_table(raw_df, latex_path)

    plot_radar(
        norm_df=norm_df,
        family_config=family_config,
        output_png=radar_png,
        output_pdf=radar_pdf,
        title=args.title,
        include_legend=not args.no_legend,
    )

    print("\n" + "=" * 120)
    print("RAW METRIC VALUES")
    print("=" * 120)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        2400,
        "display.float_format",
        "{:.5g}".format,
    ):
        print(raw_df[["Family", "Display Name"] + ALL_METRICS].to_string(index=False))

    print("\n" + "=" * 120)
    print("NORMALIZED METRIC VALUES")
    print("=" * 120)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        2400,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(norm_df[["Family", "Display Name"] + [f"{m}_norm" for m in ALL_METRICS]].to_string(index=False))

    logging.info("Saved raw CSV: %s", raw_csv)
    logging.info("Saved normalized CSV: %s", norm_csv)
    logging.info("Saved LaTeX table: %s", latex_path)
    logging.info("Saved radar PNG: %s", radar_png)
    if radar_pdf is not None:
        logging.info("Saved radar PDF: %s", radar_pdf)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate tables and figures for similarity + diversity evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--similarity_cache",
        type=str,
        default="outputs/unified_benchmark_cache.json",
        help="Cache produced by evaluate_distribution_similarity.py.",
    )
    parser.add_argument(
        "--diversity_cache",
        type=str,
        default="outputs/diversity_cache.json",
        help="Cache produced by evaluate_diversity.py.",
    )
    parser.add_argument(
        "--family_config",
        type=str,
        default=None,
        help=(
            "Optional JSON family config. If omitted, uses built-in GPT/Gemini/Qwen mapping."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="outputs/figures")

    parser.add_argument("--raw_csv_name", type=str, default="combined_raw_metrics.csv")
    parser.add_argument("--normalized_csv_name", type=str, default="combined_normalized_metrics.csv")
    parser.add_argument("--latex_name", type=str, default="combined_metrics_table.tex")
    parser.add_argument("--radar_png_name", type=str, default="paper_12metric_radar.png")
    parser.add_argument("--radar_pdf_name", type=str, default="paper_12metric_radar.pdf")

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    generate_tables_and_figures(args)


if __name__ == "__main__":
    main()