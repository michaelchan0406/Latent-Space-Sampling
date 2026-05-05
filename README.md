# Distribution-Aware Medical Benchmark Generation via Latent-Space Sampling

This repository contains anonymized code for a NeurIPS 2026 Evaluations and Datasets Track submission. The code implements a latent-space sampling pipeline for generating distribution-aware medical benchmark questions, baseline generators, evaluation metrics, downstream model evaluation, and ablations.

The repository is intended for reviewer inspection and reproduction. Full-scale runs require external datasets, model downloads, and LLM API access; those external resources are not bundled in this anonymous repository.

## Repository structure

```text
Latent Space Sampling/        Core latent-space pipeline
  classify_icd10_chapters.py  LLM-assisted ICD-10 chapter labeling
  train_question_classifier.py
  train_local_gmms.py
  full_pipeline.py

Evaluation/                   Baseline generation and benchmark evaluation
  baseline.py
  diversity_evaluation.py
  evaluate_distribution_similarity.py
  downstream_llm_evaluation.py
  make_tables_and_fitures.py

ablation study/               Ablation pipelines and quality judging
configs/                      Example configuration files
scripts/                      Minimal reproduction command templates
utils/                        Small utility scripts
requirements.txt              Python dependencies
```

## Installation

We recommend Python 3.10+ and a fresh virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For a quick syntax check:

```bash
bash scripts/00_smoke_test.sh
```

## External data and API requirements

The code expects the following external resources to be prepared locally:

- MedQuAD and PubMedQA questions for the target/reference medical-question corpus.
- A Questions-vs-Statements style dataset for training the question-form classifier.
- Local output directories for generated artifacts and benchmark outputs.
- API keys for LLM-based ICD-10 labeling, baseline generation, rewriting, and judge evaluation.

Set API keys with environment variables as needed:

```bash
export OPENAI_API_KEY="..."
export QWEN_API_KEY="..."
export GEMINI_API_KEY="..."
```

If using OpenAI-compatible proxy endpoints, also set the relevant base URL variables or pass `--base-url` arguments where supported.

## Minimal workflow

### 1. Train reusable artifacts

Train the question-form classifier and local ICD-based GMM experts:

```bash
bash scripts/01_train_artifacts.sh
```

Before running, edit the paths in the script so they point to your local data files. The main expected artifacts are:

```text
data/models/question_detector/
  resnet_qvstmt_gtr_t5_base_L2.pt
  standardize_mu_sg_and_threshold_L2.npz

data/experts/gmms/
  gmm_<cluster_name>.joblib
```

### 2. Run latent-space benchmark generation

```bash
bash scripts/02_generate_ours.sh
```

The full pipeline writes intermediate files to `outputs/full_pipeline/`, including:

```text
step1_vectors_proportional.npy        sampled latent vectors
step2_texts.jsonl                     raw vec2text decodings
step3_clean_texts.jsonl               gibberish-filtered raw clean outputs
step4_final_refined.jsonl             final LLM-rewritten questions
benchmark.csv                         CSV converted for downstream evaluation
```

### 3. Run diversity and distribution evaluations

Prepare a manifest following `configs/eval_manifest.example.csv`, then run:

```bash
bash scripts/03_evaluate.sh
```

The paper setting uses 2,000 candidate questions per method, 100 Monte Carlo runs, and 1,000 sampled questions per run.

### 4. Run downstream model evaluation

```bash
bash scripts/04_downstream_eval.sh
```

By default, the target answer-generation models are GPT-5.4, Gemini-3-Pro, Gemini-3-Flash, Qwen-3.5-Flash, and GPT-4o. The judge ensemble is GPT-5.4, Gemini-3-Flash, and Qwen-3.5-Flash.

### 5. Run the no-rewrite ablation

The **No LLM Rewrite** ablation directly evaluates the gibberish-filtered raw clean outputs from the full pipeline:

```text
outputs/full_pipeline/step3_clean_texts.jsonl
```

Run:

```bash
bash scripts/05_ablation_no_rewrite.sh
```

This corresponds to evaluating the pipeline output after gibberish filtering and before LLM clinical rewriting.

## Main scripts

### ICD-10 labeling

```bash
python "Latent Space Sampling/classify_icd10_chapters.py" \
  --sources-json ./configs/icd10_sources.example.json \
  --output-dir ./artifacts/icd10_labeled \
  --model gpt-4o-mini
```

### Question classifier training

```bash
python "Latent Space Sampling/train_question_classifier.py" \
  --csv-path ./data/questions_vs_statements.csv \
  --output-dir ./data/models/question_detector \
  --text-column doc \
  --label-column target
```

### Local GMM training

```bash
python "Latent Space Sampling/train_local_gmms.py" \
  --labeled-csv ./data/references/labeled_MedQuAD_PubMedQA_GT.csv \
  --output-dir ./data/experts/gmms \
  --text-column text \
  --chapter-column chapter_code \
  --max-components 15
```

The implementation uses up to 15 full-covariance GMM components per ICD-derived super cluster, depending on cluster size.

### Full generation pipeline

```bash
python "Latent Space Sampling/full_pipeline.py" \
  --data-root ./data \
  --output-dir ./outputs/full_pipeline \
  --target-total 21000 \
  --steps all \
  --rewrite-model gpt-5.4
```

### Convert generated JSONL to CSV

```bash
python utils/jsonl_to_benchmark_csv.py \
  --input ./outputs/full_pipeline/step4_final_refined.jsonl \
  --output ./outputs/full_pipeline/benchmark.csv \
  --text-field rewritten
```

For the no-rewrite ablation, use:

```bash
python utils/jsonl_to_benchmark_csv.py \
  --input ./outputs/full_pipeline/step3_clean_texts.jsonl \
  --output ./outputs/full_pipeline/no_rewrite_benchmark.csv \
  --text-field text
```

## Notes on reproducibility

- Large generated outputs, external datasets, and API-generated results are not bundled in this anonymous code release.
- Numerical results involving LLM rewriting, baseline generation, or LLM judging can vary with model version, API backend, and decoding settings.
- The scripts expose seeds and major sampling parameters through command-line arguments.
- The repository avoids embedding private paths, author names, institutional names, and API keys.
