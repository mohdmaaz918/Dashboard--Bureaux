# US Deployment Manifest

## Runtime

- `src/bureau_uw`
- `pyproject.toml`
- `requirements-integration.txt`
- `artifacts/us_equifax_lr_l1/model_pipeline.joblib`
- `artifacts/us_equifax_lr_l1/feature_list.json`
- `artifacts/us_equifax_lr_l1/policy.json`
- `artifacts/us_equifax_lr_l1/coefficients.csv`

## Integration

- `README.md`
- `INTEGRATION.md`
- `example_rest_api.py`

## Current US Runtime Behavior

- revolving credit limit uses trade-level summed `creditLimit` for revolving/open trades
- Equifax utilization-style attributes are normalized to proper percentage scale
- US-facing metric descriptions are aligned to current dashboard wording

## Rebuild and Audit

- `configs/us_dataset_config.yaml`
- `scripts/build_us_training_dataset.py`
- `scripts/build_us_candidate_feature_list.py`
- `scripts/prepare_us_training_labels_template.py`
- `scripts/feature_diagnostics.py`
- `scripts/train_lr_l1.py`
- `scripts/train_us_model.bat`
- `reports/us_candidate_features.csv`
- `reports/us_feature_diagnostics.csv`
- `data/processed/us_feature_schema.json`
- `artifacts/us_equifax_lr_l1/train_summary.json`
- `artifacts/us_equifax_lr_l1/cv_results.csv`
- `artifacts/us_equifax_lr_l1/scored_training.csv`
- `US dashboard info/ResponseEssentials.json`
- `US dashboard info/metrics-calculation.txt`
