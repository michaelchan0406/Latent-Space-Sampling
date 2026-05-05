#!/usr/bin/env bash
set -euo pipefail

python "Evaluation/downstream_llm_evaluation.py" \
  --benchmark-path ./outputs/full_pipeline/benchmark.csv \
  --output-dir ./outputs/downstream_eval \
  --question-column text \
  --sample-size 300 \
  --models gemini-3-flash-preview gemini-3-pro-preview gpt-4o gpt-5.4 qwen3.5-flash \
  --judge-models gpt-5.4 gemini-3-flash-preview qwen3.5-flash
