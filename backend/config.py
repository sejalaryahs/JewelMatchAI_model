import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BACKEND_DIR.parent

MODELS_DIR = PROJECT_DIR / "models"

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# MODEL 3 - ONNX CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(MODELS_DIR / "jewellery_model3.onnx"),
)

DEVICE = os.getenv("DEVICE", "cpu")

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))


# ============================================================
# DATABASE CONFIGURATION
# Keep your existing MongoDB settings here for now.
# ============================================================

# Do not change your existing database configuration,
# collection names, or GridFS settings during ONNX migration.


# ============================================================
# MATCHING CONFIGURATION
# Keep your existing matching settings here.
# ============================================================

# Do not change your existing TOP_K, similarity threshold,
# or other matching settings until matcher.py is updated.
