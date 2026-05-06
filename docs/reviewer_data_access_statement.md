# Reviewer data access statement

Anonymous artifact repository: https://anonymous.4open.science/r/Latent-Space-Sampling-BD8C/

The artifact does not bundle external public datasets. Reviewers can reproduce the data preparation steps by downloading or loading the source datasets from their original public locations:

- PubMedQA: https://pubmedqa.github.io/ and https://github.com/pubmedqa/pubmedqa
- MedQuAD: https://github.com/abachaa/MedQuAD
- Questions vs Statements Classification Dataset: https://www.kaggle.com/datasets/shahrukhkhan/questions-vs-statementsclassificationdataset

The repository provides code and configuration files for preprocessing these public sources and running the MEDALS generation/evaluation pipeline. API keys, private credentials, proprietary model endpoints, and raw copies of the external datasets are not included.

If exact reproduction requires closed-source LLM API calls, outputs may vary due to provider-side model updates. The repository documents LLM roles, prompts, and decoding parameters in `metadata/llm_usage_metadata.json` and in the supplement.
