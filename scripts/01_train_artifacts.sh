#!/usr/bin/env bash
set -euo pipefail

# Edit paths under ./data before running.
python "Latent Space Sampling/train_question_classifier.py" \
  --csv-path ./data/questions_vs_statements.csv \
  --output-dir ./data/models/question_detector \
  --text-column doc \
  --label-column target

python "Latent Space Sampling/train_local_gmms.py" \
  --labeled-csv ./data/references/labeled_MedQuAD_PubMedQA_GT.csv \
  --output-dir ./data/experts/gmms \
  --text-column text \
  --chapter-column chapter_code \
  --max-components 15
