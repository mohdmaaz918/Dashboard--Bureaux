# US Bureau Underwriting Deployment Artifacts

This folder is the US market server deployment package. It is intentionally separate from the UK package.

## Included

- `src/bureau_uw`: headless scoring package, including US Equifax JSON parsing, feature engineering, scoring, policy decisions, risk pillars, and response shaping.
- `artifacts/us_equifax_lr_l1`: US model bundle used by the current US app.
- `configs/us_dataset_config.yaml`: US training dataset configuration.
- `reports/us_candidate_features.csv` and `reports/us_feature_diagnostics.csv`: US feature selection and diagnostics.
- `US dashboard info`: US source/reference files currently kept with the app.
- `scripts`: US rebuild/training scripts.
- `example_rest_api.py`: minimal FastAPI wrapper for server deployment.

## Current US Metric Behavior

This package reflects the current US dashboard/runtime behavior:

- revolving credit limit is taken from the **trade-level summed `creditLimit`** on revolving/open trades
- Equifax utilization-style summary attributes are normalized to the correct percentage scale
  - example: `421` is interpreted as `4.21%`, not `421%`
- US-facing descriptions are aligned to Equifax / US terminology rather than UK labels
- the US summary is a mix of:
  - mapped Equifax summary attributes
  - computed trade-level fallback values

## Server Use

Install from this folder:

```bash
pip install -e .
pip install -r requirements-integration.txt
```

Set the artifact directory:

```bash
export BUREAU_UW_ARTIFACT_DIR=/path/to/Dev\ Artifacts\ -\ US/artifacts/us_equifax_lr_l1
```

For in-process scoring, use:

```python
from bureau_uw.scoring_api import score_bureau_input

result = score_bureau_input(
    json_bytes,
    "artifacts/us_equifax_lr_l1",
    input_format="json",
    bureau="equifax_us",
)
```

US input is Equifax JSON. Do not point this deployment at `artifacts/baseline_lr_l1`.
