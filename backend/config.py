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
# ENVIRONMENT VARIABLES
# ============================================================

ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

FLASK_HOST = os.getenv(
    "FLASK_HOST",
    "0.0.0.0",
)

FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

DEBUG = (
    os.getenv(
        "FLASK_DEBUG",
        "False",
    ).lower()
    == "true"
)


# ============================================================
# MONGODB CONFIGURATION
# ============================================================

MONGO_URI = os.getenv(
    "MONGO_URI",
    "",
)

# Existing MongoDB database
MONGO_DATABASE = os.getenv(
    "MONGO_DATABASE",
    "jewelmatch",
)

# Existing prototype jewellery collection
MONGO_COLLECTION = os.getenv(
    "MONGO_COLLECTION",
    "prototype",
)

# MongoDB GridFS configuration
MONGO_GRIDFS_BUCKET = os.getenv(
    "MONGO_GRIDFS_BUCKET",
    "fs",
)

# Existing embedding collections
MONGO_EMBEDDING_COLLECTION_V1 = os.getenv(
    "MONGO_EMBEDDING_COLLECTION_V1",
    "prototype_dinov2_embeddings_v1",
)

MONGO_EMBEDDING_COLLECTION_V2 = os.getenv(
    "MONGO_EMBEDDING_COLLECTION_V2",
    "prototype_dinov2_embeddings_v2",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(CHECKPOINT_DIR / "best_jewellery_model.pth"),
)

DEVICE = os.getenv(
    "DEVICE",
    "cpu",
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))


# ============================================================
# MATCHING CONFIGURATION
# ============================================================

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))


# ============================================================
# CORS CONFIGURATION
# ============================================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173",
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================


def validate_config():
    """
    Validate application configuration.
    """

    # --------------------------------------------------------
    # FLASK PORT
    # --------------------------------------------------------

    if not 1 <= FLASK_PORT <= 65535:

        raise ValueError("FLASK_PORT must be between 1 and 65535.")

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if DEVICE not in ("cpu", "cuda"):

        raise ValueError("DEVICE must be either 'cpu' or 'cuda'.")

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    if IMAGE_SIZE <= 0:

        raise ValueError("IMAGE_SIZE must be greater than zero.")

    # --------------------------------------------------------
    # IMAGE SIZE LIMIT
    # --------------------------------------------------------

    if MAX_IMAGE_SIZE_MB <= 0:

        raise ValueError("MAX_IMAGE_SIZE_MB must be greater than zero.")

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    if DEFAULT_TOP_K <= 0:

        raise ValueError("DEFAULT_TOP_K must be greater than zero.")

    if MAX_TOP_K < DEFAULT_TOP_K:

        raise ValueError("MAX_TOP_K cannot be smaller than DEFAULT_TOP_K.")

    # --------------------------------------------------------
    # MONGODB
    # --------------------------------------------------------

    if not MONGO_DATABASE:

        raise ValueError("MONGO_DATABASE cannot be empty.")

    if not MONGO_COLLECTION:

        raise ValueError("MONGO_COLLECTION cannot be empty.")


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

validate_config()


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print("JEWELLERY IMAGE TRAINING CONFIGURATION")

    print("=" * 60)

    print(
        "Project directory:",
        PROJECT_DIR,
    )

    print(
        "Backend directory:",
        BACKEND_DIR,
    )

    print(
        "Models directory:",
        MODELS_DIR,
    )

    print(
        "Checkpoint directory:",
        CHECKPOINT_DIR,
    )

    print(
        "Model checkpoint:",
        MODEL_PATH,
    )

    print(
        "Checkpoint exists:",
        Path(MODEL_PATH).is_file(),
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Image size:",
        IMAGE_SIZE,
    )

    print(
        "MongoDB database:",
        MONGO_DATABASE,
    )

    print(
        "MongoDB collection:",
        MONGO_COLLECTION,
    )

    print(
        "GridFS bucket:",
        MONGO_GRIDFS_BUCKET,
    )

    print(
        "Default top K:",
        DEFAULT_TOP_K,
    )

    print(
        "Maximum top K:",
        MAX_TOP_K,
    )

    print("=" * 60)
