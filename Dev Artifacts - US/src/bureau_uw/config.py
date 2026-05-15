from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PAID_DIR = DATA_DIR / "raw" / "paid"
RAW_NOT_PAID_DIR = DATA_DIR / "raw" / "not_paid"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
