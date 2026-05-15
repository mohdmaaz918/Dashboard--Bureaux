# US Model Artifact Bundle

Use `us_equifax_lr_l1` for the US deployment.

Required at runtime:

- `us_equifax_lr_l1/model_pipeline.joblib`
- `us_equifax_lr_l1/feature_list.json`
- `us_equifax_lr_l1/policy.json`

Additional explainability and audit files:

- `us_equifax_lr_l1/coefficients.csv`
- `us_equifax_lr_l1/cv_results.csv`
- `us_equifax_lr_l1/scored_training.csv`
- `us_equifax_lr_l1/train_summary.json`

Do not add or substitute the UK `baseline_lr_l1` model bundle in this US package.
