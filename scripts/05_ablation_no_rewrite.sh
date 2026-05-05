#!/usr/bin/env bash
set -euo pipefail

# No LLM Rewrite ablation: directly evaluate gibberish-filtered raw clean outputs.
# These are produced by the full pipeline before the rewriting stage.
python "ablation study/evaluate_quality_llm_judge.py" \
  --input-file ./outputs/full_pipeline/step3_clean_texts.jsonl \
  --output-dir ./outputs/ablation/no_llm_rewrite_quality \
  --text-field text \
  --run-name no_llm_rewrite \
  --judge-model gpt-4o-mini
