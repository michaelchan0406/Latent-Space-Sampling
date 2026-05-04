# Distribution-Aware Medical Benchmark Generation via Latent-Space Sampling

This repository contains the anonymized code for a NeurIPS 2026 Evaluations and Datasets Track submission on distribution-aware medical benchmark generation. The code implements a latent-space sampling framework for generating clinically plausible medical questions, together with baseline generators, distribution/diversity evaluations, downstream LLM evaluation, and ablation studies.

The repository is designed for review-time inspection and reproducibility. All scripts are command-line driven, avoid hard-coded private paths, and read API credentials from environment variables.

> **Anonymity note.** This repository is prepared for double-blind review. It intentionally omits author names, affiliations, personal accounts, private paths, and institution-specific metadata. Add citation, contact, license, and project homepage details only after de-anonymization if appropriate.

---

## Contents

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── icd10_sources.example.json
│   ├── eval_manifest.example.csv
│   └── family_config.example.json
├── Latent Space Sampling/
│   ├── classify_icd10_chapters.py
│   ├── train_question_classifier.py
│   ├── train_local_gmms.py
│   └── full_pipeline.py
├── Evaluation/
│   ├── baseline.py
│   ├── diversity_evaluation.py
│   ├── evaluate_distribution_similarity.py
│   ├── downstream_llm_evaluation.py
│   └── make_tables_and_fitures.py
├── ablation study/
│   ├── ablation_global_gmm.py
│   ├── ablation_gibberish_detector.py
│   ├── ablation_no_question_classifier.py
│   ├── evaluate_quality_llm_judge.py
│   └── summarize_cross_model_evaluation.py
├── scripts/
│   ├── run_generation.sh
│   ├── run_baselines.sh
│   ├── run_evaluation.sh
│   └── run_ablation.sh
└── examples/
    ├── sample_questions.jsonl
    └── sample_outputs/
```

Two repository directories contain spaces. Quote paths when running scripts:

```bash
python "Latent Space Sampling/full_pipeline.py" --help
python "ablation study/evaluate_quality_llm_judge.py" --help
```

---

## High-level pipeline

The main method constructs a synthetic medical-question benchmark through the following stages:

1. **ICD-10 chapter labeling** of source medical questions.
2. **ICD-10-based local GMM training** over normalized `sentence-transformers/gtr-t5-base` question embeddings.
3. **Proportional latent-space sampling** from local GMM experts according to the empirical clinical-topic distribution.
4. **Latent question-form filtering** using a residual MLP trained on question-vs-statement data.
5. **vec2text embedding inversion** to decode accepted latent vectors into text.
6. **Gibberish filtering** to remove linguistically invalid decodings.
7. **LLM-based clinical rewriting** to obtain fluent, clinically plausible, and answerable medical questions.

The repository also includes prompt-based, instruction-generation, retrieval-grounded, and ablation baselines.

---

## Environment setup

Recommended Python version: **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is not used, the core dependencies are:

```bash
python -m pip install \
  numpy pandas tqdm scikit-learn scipy matplotlib joblib \
  torch transformers openai datasets nltk
```

Optional components:

```bash
# Needed for Self-Instruct and WizardLM/Evol-Instruct baselines.
python -m pip install distilabel

# Needed for embedding inversion in the full pipeline and ablations.
python -m pip install vec2text
```

A CUDA-capable GPU is recommended for embedding extraction and vec2text inversion. Some evaluation scripts can run on CPU but may be slow.

---

## API keys and model endpoints

LLM-based scripts use OpenAI-compatible chat-completion clients. Set only the keys required for the models you run:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export QWEN_API_KEY="..."
```

Optional custom endpoints:

```bash
export OPENAI_BASE_URL="..."
export GEMINI_BASE_URL="..."
export QWEN_BASE_URL="..."
```

LLM outputs can vary across provider-side model updates. For exact post hoc analysis, preserve generated JSONL files and evaluation caches.

---

## Expected data layout

The full pipeline expects the following default structure under `--data-root`:

```text
data/
├── experts/
│   └── gmms/
│       └── gmm_<cluster_name>.joblib
├── models/
│   └── question_detector/
│       ├── resnet_qvstmt_gtr_t5_base_L2.pt
│       └── standardize_mu_sg_and_threshold_L2.npz
├── references/
│   └── labeled_MedQuAD_PubMedQA_GT.csv
└── embeddings_gtr_t5_base/
    ├── medquad_questions.gtr_t5_base.float32.npy
    └── pubmed_qa_unlabeled.gtr_t5_base.float32.npy
```

Expected tabular formats:

- ICD-10-labeled source CSV: `text`, `chapter_code`
- Question-vs-statement classifier CSV: `doc`, `target`
- Evaluation manifest CSV: `method`, `path`, and optionally `label_path`

`HUGGINGFACE_MIXED` is treated as a special manifest path for the combined MedQuAD + PubMedQA ground-truth reference pool.

---

## Reproducing the main method

### 1. ICD-10 chapter labeling

Create `configs/icd10_sources.example.json` or an equivalent source manifest:

```json
[
  {
    "name": "MedQuAD_PubMedQA_GT",
    "format": "huggingface_mixed"
  },
  {
    "name": "full_pipeline",
    "path": "./outputs/full_pipeline/step4_final_refined.jsonl",
    "format": "jsonl",
    "text_column": "rewritten"
  }
]
```

Run:

```bash
python "Latent Space Sampling/classify_icd10_chapters.py" \
  --sources-json ./configs/icd10_sources.example.json \
  --output-dir ./data/references \
  --model gpt-4o-mini \
  --batch-size 10 \
  --chunk-size 500 \
  --seed 2026
```

Main outputs:

```text
data/references/labeled_<source_name>.csv
data/references/icd10_classification_summary.csv
```

For the default pipeline layout, the ground-truth label file should be available as:

```text
data/references/labeled_MedQuAD_PubMedQA_GT.csv
```

### 2. Train local ICD-10 GMM experts

```bash
python "Latent Space Sampling/train_local_gmms.py" \
  --labeled-csv ./data/references/labeled_MedQuAD_PubMedQA_GT.csv \
  --output-dir ./data/experts \
  --text-column text \
  --chapter-column chapter_code \
  --encoder-model sentence-transformers/gtr-t5-base \
  --embedding-batch-size 256 \
  --max-length 128 \
  --max-components 15 \
  --seed 2026 \
  --save-cluster-embeddings
```

Main outputs:

```text
data/experts/gmms/gmm_<cluster_name>.joblib
data/experts/local_gmm_training_summary.csv
data/experts/local_gmm_training_metadata.json
```

The implementation fits up to 15 full-covariance GMM components per ICD-10-derived super cluster, subject to available samples.

### 3. Train the latent question-form classifier

```bash
python "Latent Space Sampling/train_question_classifier.py" \
  --csv-path ./data/questions_vs_statements_v1.0.csv \
  --output-dir ./data/models/question_detector \
  --text-column doc \
  --label-column target \
  --embedding-cache ./data/models/question_detector/qvstmt_gtr_t5_base_embeddings_L2.npy \
  --seed 2026 \
  --split-seed 2025 \
  --batch-size 512 \
  --embedding-batch-size 256 \
  --max-epochs 40
```

Main outputs:

```text
data/models/question_detector/resnet_qvstmt_gtr_t5_base_L2.pt
data/models/question_detector/standardize_mu_sg_and_threshold_L2.npz
```

### 4. Run latent-space benchmark generation

```bash
python "Latent Space Sampling/full_pipeline.py" \
  --data-root ./data \
  --output-dir ./outputs/full_pipeline \
  --target-total 21000 \
  --steps all \
  --rewrite-model gpt-5.4 \
  --seed 42
```

Main outputs:

```text
outputs/full_pipeline/step1_vectors_proportional.npy
outputs/full_pipeline/step1_vectors_proportional_meta.jsonl
outputs/full_pipeline/step1_sampling_quotas.json
outputs/full_pipeline/step2_texts.jsonl
outputs/full_pipeline/step3_clean_texts.jsonl
outputs/full_pipeline/step4_final_refined.jsonl
outputs/full_pipeline/step4_refined.done.txt
```

The final benchmark questions are stored in `step4_final_refined.jsonl`, typically in the `rewritten` field.

If downstream evaluation requires CSV input, convert the final JSONL to CSV with the appropriate text field:

```bash
python - <<'PY'
import json
import pandas as pd

src = "./outputs/full_pipeline/step4_final_refined.jsonl"
dst = "./outputs/full_pipeline/final_benchmark.csv"
rows = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        q = obj.get("rewritten") or obj.get("text") or obj.get("question")
        if q:
            rows.append({"text": q})
pd.DataFrame(rows).to_csv(dst, index=False)
print(f"Wrote {len(rows)} rows to {dst}")
PY
```

---

## Baseline generation

`Evaluation/baseline.py` implements:

- `m1`: iterative one-question-per-call prompting,
- `m2`: batch synthetic generation,
- `self_instruct`: Distilabel Self-Instruct,
- `wizardlm`: WizardLM-style Evol-Instruct,
- `med_rag`: retrieval-grounded medical question generation.

Example:

```bash
python "Evaluation/baseline.py" \
  --baselines m1,m2,self_instruct,wizardlm,med_rag \
  --models gpt-5.4,gemini-3-flash-preview,qwen3.5-flash \
  --n 2000 \
  --output_dir ./outputs/baselines \
  --resume
```

Outputs are JSONL files under `outputs/baselines/`, unless `--output_file` is provided.

---

## Evaluation

### 1. Evaluation manifest

Create `configs/eval_manifest.example.csv`:

```csv
method,path,label_path
MedQuAD+PubMedQA (GT),HUGGINGFACE_MIXED,./data/references/labeled_MedQuAD_PubMedQA_GT.csv
Latent Space Sampling GPT-5.4,./outputs/full_pipeline/step4_final_refined.jsonl,./data/references/labeled_full_pipeline.csv
M1 GPT-5.4,./outputs/baselines/m1__gpt-5.4.jsonl,./data/references/labeled_m1_gpt5.4.csv
M2 GPT-5.4,./outputs/baselines/m2__gpt-5.4.jsonl,./data/references/labeled_m2_gpt5.4.csv
Med-RAG GPT-5.4,./outputs/baselines/med_rag__gpt-5.4.jsonl,./data/references/labeled_medrag_gpt5.4.csv
Self-Instruct GPT-5.4,./outputs/baselines/self_instruct__gpt-5.4.jsonl,./data/references/labeled_self_instruct_gpt5.4.csv
WizardLM GPT-5.4,./outputs/baselines/wizardlm__gpt-5.4.jsonl,./data/references/labeled_wizardlm_gpt5.4.csv
```

The label files can be generated using `classify_icd10_chapters.py`.

### 2. Diversity evaluation

This reproduces the paper setting of 100 Monte Carlo runs with 1000 sampled questions per run from pools of up to 2000 valid questions.

```bash
python "Evaluation/diversity_evaluation.py" \
  --manifest ./configs/eval_manifest.example.csv \
  --cache_file ./outputs/diversity_cache.json \
  --csv_file ./outputs/diversity_results.csv \
  --gt_name "MedQuAD+PubMedQA (GT)" \
  --pool_size 2000 \
  --sample_size 1000 \
  --runs 100
```

Reported metrics include Dist-1, Dist-2, Self-BLEU, V-EE, topic coverage, topic entropy, Vendi score, gzip ratio, and mean pairwise embedding distance.

### 3. Distribution similarity evaluation

This also uses the paper setting of 100 Monte Carlo runs with 1000 sampled questions per run.

```bash
python "Evaluation/evaluate_distribution_similarity.py" \
  --manifest ./configs/eval_manifest.example.csv \
  --cache_file ./outputs/unified_benchmark_cache.json \
  --csv_file ./outputs/distribution_similarity_results.csv \
  --gt_name "MedQuAD+PubMedQA (GT)" \
  --pool_size 2000 \
  --sample_size 1000 \
  --min_samples 1000 \
  --runs 100
```

Reported metrics include MMD, Frechet embedding distance, and KL divergence over ICD-10 chapter distributions.

### 4. Downstream LLM evaluation

The downstream study samples 300 benchmark questions, asks target models to answer them, and evaluates responses with LLM judges on accuracy, completeness, and safety.

```bash
python "Evaluation/downstream_llm_evaluation.py" \
  --benchmark-path ./outputs/full_pipeline/final_benchmark.csv \
  --output-dir ./outputs/downstream_eval \
  --question-column text \
  --sample-size 300 \
  --seed 2026 \
  --models gemini-3-flash-preview gemini-3-pro-preview gpt-4o gpt-5.4 qwen3.5-flash \
  --judge-models gpt-5.4 gemini-3-flash-preview qwen3.5-flash
```

The judge ensemble used for the main downstream evaluation is:

```text
gpt-5.4
gemini-3-flash-preview
qwen3.5-flash
```

### 5. Tables and figures

```bash
python "Evaluation/make_tables_and_fitures.py" \
  --similarity_cache ./outputs/unified_benchmark_cache.json \
  --diversity_cache ./outputs/diversity_cache.json \
  --output_dir ./outputs/figures \
  --save_pdf
```

Main outputs:

```text
outputs/figures/combined_raw_metrics.csv
outputs/figures/combined_normalized_metrics.csv
outputs/figures/combined_metrics_table.tex
outputs/figures/paper_12metric_radar.png
outputs/figures/paper_12metric_radar.pdf
```

---

## Ablation studies

### 1. Global GMM instead of ICD-10 local experts

```bash
python "ablation study/ablation_global_gmm.py" \
  --data-root ./data \
  --output-dir ./outputs/ablation_global_gmm \
  --steps all \
  --question-model-dir ./data/models/question_detector \
  --seed 42
```

This ablation fits or loads a single global GMM over reference embeddings, then applies question filtering, vec2text decoding, gibberish filtering, and LLM rewriting.

### 2. No latent question classifier

```bash
python "ablation study/ablation_no_question_classifier.py" \
  --data-root ./data \
  --output-dir ./outputs/ablation_no_question_classifier \
  --steps all \
  --seed 42
```

This ablation keeps ICD-10 local GMM sampling but removes the latent question-form classifier before vec2text decoding.

### 3. No gibberish detector

```bash
python "ablation study/ablation_gibberish_detector.py" \
  --data-root ./data \
  --output-dir ./outputs/ablation_no_gibberish \
  --steps all \
  --question-model-dir ./data/models/question_detector \
  --seed 42
```

This ablation keeps local GMM sampling and latent question filtering but removes gibberish filtering before the LLM rewriting stage.

### 4. No LLM Rewrite

The **No LLM Rewrite** ablation is evaluated directly on the raw clean outputs after gibberish filtering, before the LLM clinical rewriting stage. In the full pipeline, these outputs are stored in:

```text
outputs/full_pipeline/step3_clean_texts.jsonl
```

Run the quality judge on `step3_clean_texts.jsonl` using the raw text field:

```bash
python "ablation study/evaluate_quality_llm_judge.py" \
  --input-file ./outputs/full_pipeline/step3_clean_texts.jsonl \
  --output-dir ./outputs/quality_eval/no_llm_rewrite \
  --text-field text \
  --run-name no_llm_rewrite \
  --judge-model gpt-4o-mini \
  --sample-size 500 \
  --seed 2026
```

This ablation measures the usability of vec2text-decoded questions after gibberish filtering but before LLM-based clinical rewriting. It is intended to isolate the contribution of the rewriting stage.

### 5. Full pipeline quality evaluation

For comparison, evaluate the final rewritten benchmark:

```bash
python "ablation study/evaluate_quality_llm_judge.py" \
  --input-file ./outputs/full_pipeline/step4_final_refined.jsonl \
  --output-dir ./outputs/quality_eval/full_pipeline \
  --text-field rewritten \
  --run-name full_pipeline \
  --judge-model gpt-4o-mini \
  --sample-size 500 \
  --seed 2026
```

Quality-judge outputs:

```text
<run-name>_judge_results.jsonl
<run-name>_judge_results.csv
<run-name>_judge_summary.json
```

---

## Reproducibility notes

- Use fixed `--seed` values for all reported runs.
- Preserve generated JSONL files, CSV outputs, and evaluation caches for exact post hoc analysis.
- Use `--resume` where available for long-running API-based generation.
- Use `--overwrite`, `--force`, or `--force_recompute` only when intentionally rerunning stages from scratch.
- LLM-based rewriting and judging may vary because model providers can update hosted models over time.
- For exact reproduction of paper tables, use the same generated benchmark files and cached evaluation outputs used in the submission.

---

## Troubleshooting

### `vec2text` import or model-loading errors

Install `vec2text` before running `full_pipeline.py` or ablation scripts that perform embedding inversion. If only evaluating existing JSONL outputs, `vec2text` is not required.

### Missing question-detector files

Ensure the following files exist:

```text
resnet_qvstmt_gtr_t5_base_L2.pt
standardize_mu_sg_and_threshold_L2.npz
```

Useful flags:

```bash
--question-model-dir ./data/models/question_detector
--question-model-pt ./data/models/question_detector/resnet_qvstmt_gtr_t5_base_L2.pt
--question-stats-npz ./data/models/question_detector/standardize_mu_sg_and_threshold_L2.npz
```

### Missing ICD-10 labels during similarity evaluation

`evaluate_distribution_similarity.py` expects labels for KL divergence. Provide `label_path` in the evaluation manifest or use a label directory containing the expected labeled CSV files.

### Slow embedding or inversion stages

Use GPU execution when available:

```bash
--device cuda
```

For debug runs, reduce sample counts:

```bash
--target-total 200
--pool_size 200
--sample_size 100
--runs 3
```

---

## Pre-submission code-release checklist

Before submitting the Code URL for review, verify that:

- [ ] The repository is anonymized and does not contain author names, affiliations, private paths, email addresses, or personal access tokens.
- [ ] The repository is accessible without login at submission time.
- [ ] `README.md` documents installation, data preparation, generation, baselines, evaluation, and ablations.
- [ ] `requirements.txt` or an equivalent environment file is included.
- [ ] Example config files are included under `configs/`.
- [ ] Small sample outputs are included under `examples/` if full outputs cannot be released at submission time.
- [ ] The main scripts run with `--help`.
- [ ] The repository does not include `.DS_Store`, `__MACOSX`, API keys, cache files with private metadata, or large accidental artifacts.

Useful cleanup commands:

```bash
find . -name ".DS_Store" -delete
rm -rf __MACOSX
python -m py_compile $(find . -name "*.py")
```

---

## Citation

Citation information should be added after de-anonymization.

```bibtex
@inproceedings{anonymous2026latentmedicalbenchmark,
  title     = {Distribution-Aware LLM Evaluation via Latent Space Sampling},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
