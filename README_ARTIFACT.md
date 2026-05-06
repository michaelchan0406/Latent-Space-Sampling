# MEDALS artifact metadata additions

Anonymous artifact URL: https://anonymous.4open.science/r/Latent-Space-Sampling-BD8C/

These files document the MEDALS code and evaluation-method artifact. They are intended for an ED-track submission framed as a benchmark-generation/evaluation framework built on existing public datasets, rather than as a new static dataset release.

## Intended submission framing

- The submission releases code, prompts, configuration files, and evaluation scripts.
- The submission uses PubMedQA, MedQuAD, and the Kaggle Questions vs Statements Classification dataset as external public sources.
- The artifact does not bundle, re-host, or claim ownership of those public datasets.
- The artifact does not need to be presented as a new dataset contribution unless you also release a fixed generated question file such as `medals_questions.jsonl` as a central contribution.

## Files in this folder

```text
metadata/source_datasets.json
metadata/responsible_ai_statement.md
metadata/llm_usage_metadata.json
metadata/artifact_manifest.json
metadata/croissant_optional_artifact.json
docs/openreview_artifact_statement.md
docs/reviewer_data_access_statement.md
checks/anonymity_checklist.md
```

## Recommended placement

Copy the contents of this folder into the root of the anonymous artifact repository, preserving the directory structure. These files are additive and should not require changes to the core MEDALS code.

## Before submission

1. Confirm the anonymous artifact URL and code license are provided in the OpenReview submission and repository.
2. Confirm the source dataset licenses from the original pages.
3. Remove older metadata files with less polished names if they are already in the repository.
4. Run an anonymity scan before updating the anonymous repository.

## Important boundary

If you later release a fixed generated benchmark file as a new dataset, switch to the dataset-oriented hosting workflow and provide a validated Croissant file for that dataset. This package is for the method/code-artifact framing.
