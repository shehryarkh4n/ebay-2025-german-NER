from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BASE_MODEL_NAME = "microsoft/deberta-v3-large"  # won over xlm-roberta-base
WEAK_MODEL_DIR = MODELS_DIR / "deberta-improved-weak-ner-mk-2"
ENSEMBLE_MODEL_PREFIX = "deberta-ner-ensemble-seed"  # used for output dirs

MAX_SEQ_LEN = 256  # run diags on this, 192 worked last time
NUM_WORKERS = 16
