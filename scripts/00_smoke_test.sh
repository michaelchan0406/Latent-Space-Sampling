#!/usr/bin/env bash
set -euo pipefail
python -m py_compile \
  "Latent Space Sampling/classify_icd10_chapters.py" \
  "Latent Space Sampling/train_question_classifier.py" \
  "Latent Space Sampling/train_local_gmms.py" \
  "Latent Space Sampling/full_pipeline.py" \
  "Evaluation/baseline.py" \
  "Evaluation/diversity_evaluation.py" \
  "Evaluation/evaluate_distribution_similarity.py" \
  "Evaluation/downstream_llm_evaluation.py" \
  "ablation study/ablation_no_question_classifier.py" \
  "ablation study/ablation_gibberish_detector.py" \
  "ablation study/ablation_global_gmm.py" \
  "ablation study/evaluate_quality_llm_judge.py" \
  "utils/jsonl_to_benchmark_csv.py"
