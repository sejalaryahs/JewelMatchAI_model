# # ============================================================
# # JEWELMATCH AI - MONGODB CONNECTION
# # ============================================================

# import os
# from pathlib import Path

# import certifi
# from dotenv import load_dotenv
# from pymongo import MongoClient


# # ============================================================
# # PROJECT ROOT
# # ============================================================

# PROJECT_ROOT = (
#     Path(__file__)
#     .resolve()
#     .parents[2]
# )


# # ============================================================
# # ENVIRONMENT FILE
# # ============================================================

# ENV_FILE = (
#     PROJECT_ROOT
#     / ".env"
# )


# # ============================================================
# # LOAD ENVIRONMENT VARIABLES
# # ============================================================

# load_dotenv(
#     ENV_FILE,
#     override=True,
# )


# # ============================================================
# # MONGODB URI
# # ============================================================

# MONGO_URI = os.environ.get(
#     "MONGO_URI"
# )


# if not MONGO_URI:

#     raise RuntimeError(
#         "MONGO_URI is not configured. "
#         f"Expected it in: {ENV_FILE}"
#     )


# # ============================================================
# # DATABASE
# # ============================================================

# DATABASE_NAME = "jewelmatch"


# # ============================================================
# # MONGODB CLIENT
# # ============================================================

# client = MongoClient(

#     MONGO_URI,

#     tls=True,

#     tlsCAFile=certifi.where(),

#     serverSelectionTimeoutMS=10000,

#     connectTimeoutMS=20000,

#     socketTimeoutMS=20000,

#     maxPoolSize=10,

#     minPoolSize=1,

#     retryWrites=True,

#     retryReads=True,
# )


# # ============================================================
# # DATABASE OBJECT
# # ============================================================

# db = client[
#     DATABASE_NAME
# ]


# # ============================================================
# # EXISTING JEWELLERY COLLECTION
# #
# # This helper is retained so the connection structure remains
# # compatible with the matcher.
# # ============================================================

# jewellery_collection = (
#     db[
#         "jewellery"
#     ]
# )


# # ============================================================
# # GET JEWELLERY COLLECTION
# # ============================================================

# def get_jewellery_collection():

#     return jewellery_collection


# # ============================================================
# # GET DATABASE
# # ============================================================

# def get_database():

#     return db


# # ============================================================
# # TEST CONNECTION
# # ============================================================

# def test_mongodb_connection():

#     try:

#         client.admin.command(
#             "ping"
#         )

#         print(
#             "[MONGODB] Connection successful."
#         )

#         return True

#     except Exception as exc:

#         print(
#             "[MONGODB] Connection failed:"
#         )

#         print(
#             repr(exc)
#         )

#         return False

##################################################################
##################################################################


# ============================================================
# JEWELMATCH AI - MONGODB CONNECTION
# ============================================================

import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)


# ============================================================
# MONGODB URI
# ============================================================

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not configured. " f"Expected it in: {ENV_FILE}")


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_NAME = "jewelmatch"

GOLD_COLLECTION_NAME = "gold"
PROTOTYPE_COLLECTION_NAME = "prototype"
JEWELLERY_COLLECTION_NAME = "jewellery"


# ============================================================
# MONGODB CLIENT
# ============================================================

client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=20000,
    socketTimeoutMS=20000,
    maxPoolSize=10,
    minPoolSize=1,
    retryWrites=True,
    retryReads=True,
)


# ============================================================
# DATABASE OBJECT
# ============================================================

db = client[DATABASE_NAME]


# ============================================================
# COLLECTION REFERENCES
# ============================================================

# Keep this existing helper for compatibility with code that
# expects the original jewellery collection.
jewellery_collection = db[JEWELLERY_COLLECTION_NAME]

# Catalogue collections used by the existing matcher.
gold_collection = db[GOLD_COLLECTION_NAME]
prototype_collection = db[PROTOTYPE_COLLECTION_NAME]


# ============================================================
# EXISTING COMPATIBILITY HELPER
# ============================================================


def get_jewellery_collection():
    """Return the original jewellery collection."""
    return jewellery_collection


# ============================================================
# DATABASE ACCESS HELPERS
# ============================================================


def get_database():
    """Return the jewelmatch database."""
    return db


def get_gold_collection():
    """Return the gold catalogue collection."""
    return gold_collection


def get_prototype_collection():
    """Return the prototype catalogue collection."""
    return prototype_collection


def get_catalogue_collection(source):
    """
    Return the requested catalogue collection.

    Args:
        source: 'gold' or 'prototype'

    Raises:
        ValueError: If the source is unsupported.
    """
    source = str(source).strip().lower()

    collections = {
        "gold": gold_collection,
        "prototype": prototype_collection,
    }

    if source not in collections:
        raise ValueError("Invalid catalogue source. " "Expected 'gold' or 'prototype'.")

    return collections[source]


# ============================================================
# CONNECTION TEST
# ============================================================


def test_mongodb_connection():
    """Test connectivity and report catalogue collection counts."""
    try:
        client.admin.command("ping")

        print("[MONGODB] Connection successful.")
        print(f"[MONGODB] Database: {DATABASE_NAME}")
        print(f"[MONGODB] Gold records: " f"{gold_collection.count_documents({})}")
        print(
            f"[MONGODB] Prototype records: "
            f"{prototype_collection.count_documents({})}"
        )
        print(
            f"[MONGODB] Jewellery records: "
            f"{jewellery_collection.count_documents({})}"
        )

        return True

    except Exception as exc:
        print("[MONGODB] Connection failed:")
        print(repr(exc))
        return False


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    test_mongodb_connection()
