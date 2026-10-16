"""
Project-root-relative paths.

All runtime file locations are resolved from this file's location so the
application works regardless of the current working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DB_PATH = DATA_DIR / "rehab.db"
DB_BACKUP_DIR = DATA_DIR / "backups"

LSTM_MODEL_PATH = (
    OUTPUTS_DIR / "compensation_model_lstm" / "compensation_lstm_model.pt"
)

SESSION_LOG_DIR = OUTPUTS_DIR / "session_logs"
