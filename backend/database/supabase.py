import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# SUPABASE CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")

SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

SUPABASE_STORAGE_BUCKET = os.getenv(
    "SUPABASE_STORAGE_BUCKET",
    "jewellery-images",
)

TABLE_NAME = "jewellery"


# ============================================================
# SUPABASE CLIENT
# ============================================================


def get_supabase_client() -> Client:

    if not SUPABASE_URL:

        raise RuntimeError("SUPABASE_URL is missing from the .env file.")

    if not SUPABASE_SECRET_KEY:

        raise RuntimeError("SUPABASE_SECRET_KEY is missing from the .env file.")

    return create_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY,
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

supabase: Client = get_supabase_client()


# ============================================================
# STORAGE BUCKET
# ============================================================


def get_storage_bucket():

    return SUPABASE_STORAGE_BUCKET


# ============================================================
# PUBLIC IMAGE URL
# ============================================================


def get_public_image_url(image_path):

    if not image_path:

        return None

    response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(
        image_path
    )

    if isinstance(response, str):

        return response

    if isinstance(response, dict):

        return response.get("publicUrl") or response.get("publicURL")

    return getattr(
        response,
        "public_url",
        None,
    )


# ============================================================
# CONNECTION TEST
# ============================================================


def test_supabase_connection():

    try:

        response = supabase.table(TABLE_NAME).select("id").limit(1).execute()

        print("Supabase connection successful.")

        return {
            "success": True,
            "message": "Supabase connected successfully.",
        }

    except Exception as error:

        print(
            "Supabase connection failed:",
            error,
        )

        return {
            "success": False,
            "message": str(error),
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print("SUPABASE CONNECTION TEST")

    print("=" * 60)

    print("Project root:", PROJECT_ROOT)

    print("Environment file:", ENV_FILE)

    print("Environment file exists:", ENV_FILE.exists())

    print("Storage bucket:", SUPABASE_STORAGE_BUCKET)

    print()

    result = test_supabase_connection()

    print()

    print(result)
