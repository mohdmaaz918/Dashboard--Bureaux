# US Market Integration Guide

This package runs the same US scoring logic as the local US app, without requiring Streamlit on the server.

## Important US Runtime Notes

The current US runtime intentionally mixes:

- mapped Equifax summary attributes
- computed trade-level fallback values

This is expected behavior.

In particular:

- `attr__summary.indebt.totallimitsrevolve` now uses the **trade-level summed revolving/open `creditLimit`**
- `attr__summary.indebt.balancelimitratiorevolve` is treated as an **Equifax revolving utilization summary percentage**
- utilization-style Equifax values are normalized to the correct display scale
  - example: `421` becomes `4.21%`

This means some adjacent summary metrics may come from different sources by design.

## Required Artifact Directory

Use only:

```text
artifacts/us_equifax_lr_l1
```

Required files:

- `model_pipeline.joblib`
- `feature_list.json`
- `policy.json`
- `coefficients.csv`
- `train_summary.json`

## Python API

```python
from pathlib import Path
from bureau_uw.scoring_api import load_artifacts, score_bureau_input

artifact_dir = Path("artifacts/us_equifax_lr_l1")
pipe, feature_list, coef_df, policy = load_artifacts(artifact_dir)

with open("applicant.json", "rb") as f:
    json_bytes = f.read()

result = score_bureau_input(
    json_bytes,
    artifact_dir,
    input_format="json",
    bureau="equifax_us",
    customer_name="Jane Doe",
    loan_number="LN-12345",
    pipe=pipe,
    feature_list=feature_list,
    coef_df=coef_df,
    policy=policy,
)
```

## Response

The result is JSON serializable and includes:

- `score_0_100`
- `p_paid`
- `tier`
- `tier_guidance`
- `policy_decision`
- `risk_pillars`
- `key_metrics`
- `top_drivers`
- `export_payload`

`key_metrics` and `top_drivers` will include the current US-facing descriptions, including:

- `Computed revolving credit limit (trade-level)`
- `Computed revolving balance (trade-level)`
- `Equifax revolving utilization summary`
- US / Equifax wording instead of UK-oriented labels like `SHARE`

## Rebuild Notes

US training/rebuild scripts are in `scripts`, using `configs/us_dataset_config.yaml`.

This folder does not contain the UK TransUnion model bundle. Keep the UK deployment on `Dev Artifacts - UK`.
