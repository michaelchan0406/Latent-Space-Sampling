#!/usr/bin/env bash
set -euo pipefail

# Requires local GMMs, question detector artifacts, vec2text dependencies, and OPENAI_API_KEY for rewriting.
python "Latent Space Sampling/full_pipeline.py" \
  --data-root ./data \
  --output-dir ./outputs/full_pipeline \
  --target-total 21000 \
  --steps all \
  --rewrite-model gpt-5.4

python utils/jsonl_to_benchmark_csv.py \
  --input ./outputs/full_pipeline/step4_final_refined.jsonl \
  --output ./outputs/full_pipeline/benchmark.csv \
  --text-field rewritten \
  --output-column text
