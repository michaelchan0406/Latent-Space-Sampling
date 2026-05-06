# Responsible AI statement for the MEDALS artifact

## Artifact status

This submission is framed as a benchmark-generation and evaluation-method artifact, not as a new static dataset release. The released artifact consists of code, prompts, configuration files, metadata, and evaluation scripts for MEDALS. It uses existing public datasets as external sources: PubMedQA, MedQuAD, and the Kaggle Questions vs Statements Classification dataset.

These source datasets are not included in the artifact and are not claimed as new contributions. Users should obtain them from their original public locations and follow the source datasets' own licenses and terms.

## Limitations

MEDALS estimates and samples from an empirical medical-question distribution derived from public medical QA corpora. This empirical distribution should not be interpreted as the unobserved real-world distribution of clinical questions, patient encounters, hospital workflows, or physician decision-making. The released code may generate synthetic medical questions, but those questions are intended for research evaluation under documented assumptions only.

## Biases

The method inherits topic, style, language, and source biases from the public corpora used at runtime. PubMedQA emphasizes biomedical research abstracts and yes/no/maybe research questions. MedQuAD emphasizes NIH and related medical information websites and has consumer-health and website-selection biases. The question-form classifier inherits linguistic biases from the Kaggle Questions vs Statements dataset and its source corpora. Additional bias may be introduced by ICD-based super-cluster mapping, embedding models, Gaussian mixture modeling, vec2text decoding, LLM rewriting, LLM judging, and physician-evaluation sampling.

## Personal and sensitive information

The released artifact is not intended to include private patient records, protected health information, or human-subject clinical notes. However, the method operates in the health domain and can generate medical questions. Generated content should be treated as sensitive-domain evaluation material, not as medical advice.

## Recommended use cases

Recommended uses include medical LLM evaluation research, study of benchmark saturation, evaluation-methodology research, synthetic benchmark generation research, diversity/distribution-alignment analysis, and reproducibility of the MEDALS experiments under documented assumptions.

## Non-recommended use cases

Do not use MEDALS outputs for clinical diagnosis, triage, treatment recommendations, patient-facing medical advice, physician replacement, medical education without expert review, or claims of real-world clinical deployment safety. Do not claim that generated questions represent the true distribution of clinical practice.

## Social impact

Potential benefits include more transparent and distribution-aware medical LLM evaluation and better analysis of benchmark saturation. Potential risks include overclaiming clinical ability, misuse of generated questions as medical guidance, and propagation of biases from the source corpora and LLM-based rewriting/judging stages. Mitigations include explicit intended-use restrictions, clear source-dataset provenance, physician evaluation where applicable, and reporting of limitations.

## Synthetic data status

The artifact itself is code and metadata, not a static synthetic dataset. The MEDALS pipeline can generate synthetic medical benchmark questions. If a future version releases a fixed generated question set as a contribution, that release should be treated as a synthetic dataset and documented/hosted under the applicable dataset-hosting and metadata requirements.

## External sources

See `metadata/source_datasets.json` for the external source datasets, original URLs, licenses, and roles in the MEDALS pipeline.
