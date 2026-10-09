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

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# ENVIRONMENT FILE
# ============================================================

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(
    ENV_FILE,
    override=True,
)


# ============================================================
# MONGODB URI
# ============================================================

MONGO_URI = os.environ.get(
    "MONGO_URI"
)


if not MONGO_URI:

    raise RuntimeError(
        "MONGO_URI is not configured. "
        f"Expected it in: {ENV_FILE}"
    )


# ============================================================
# DATABASE
# ============================================================

DATABASE_NAME = "jewelmatch"


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

db = client[
    DATABASE_NAME
]


# ============================================================
# EXISTING JEWELLERY COLLECTION
#
# This helper is retained so the connection structure remains
# compatible with the matcher.
# ============================================================

jewellery_collection = (
    db[
        "jewellery"
    ]
)


# ============================================================
# GET JEWELLERY COLLECTION
# ============================================================

def get_jewellery_collection():

    return jewellery_collection


# ============================================================
# GET DATABASE
# ============================================================

def get_database():

    return db


# ============================================================
# TEST CONNECTION
# ============================================================

def test_mongodb_connection():

    try:

        client.admin.command(
            "ping"
        )

        print(
            "[MONGODB] Connection successful."
        )

        return True

    except Exception as exc:

        print(
            "[MONGODB] Connection failed:"
        )

        print(
            repr(exc)
        )

        return False