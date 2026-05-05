#!/usr/bin/env bash
set -euo pipefail

python "Evaluation/diversity_evaluation.py" \
  --manifest ./configs/eval_manifest.csv \
  --pool_size 2000 \
  --sample_size 1000 \
  --runs 100 \
  --min_samples 1000 \
  --csv_file ./outputs/diversity_results.csv

python "Evaluation/evaluate_distribution_similarity.py" \
  --manifest ./configs/eval_manifest.csv \
  --pool_size 2000 \
  --sample_size 1000 \
  --runs 100 \
  --min_samples 1000 \
  --csv_file ./outputs/distribution_similarity_results.csv
