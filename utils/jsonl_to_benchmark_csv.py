#!/usr/bin/env python3
"""Convert generated JSONL benchmark questions to a simple CSV with one text column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL questions to benchmark CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file.")
    parser.add_argument(
        "--text-field",
        type=str,
        default="rewritten",
        help="JSON field containing the question text. Use 'text' for step3_clean_texts.jsonl.",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="text",
        help="Name of the output CSV question column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get(args.text_field)
            if isinstance(text, str) and text.strip():
                rows.append({args.output_column: text.strip()})
            else:
                # step4 files may store the rewritten field inside a nested response object.
                response = obj.get("response")
                if isinstance(response, dict):
                    nested = response.get(args.text_field)
                    if isinstance(nested, str) and nested.strip():
                        rows.append({args.output_column: nested.strip()})

    if not rows:
        raise ValueError(f"No valid rows found in {args.input} using text field '{args.text_field}'.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} questions to {args.output}")


if __name__ == "__main__":
    main()
