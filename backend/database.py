from pymongo import MongoClient
from pymongo.errors import PyMongoError

from gridfs import GridFS

from .config import (
    MONGO_URI,
    MONGO_DATABASE,
    MONGO_COLLECTION,
    MONGO_GRIDFS_BUCKET,
)

# ============================================================
# MONGODB CONNECTION
# ============================================================

_client = None
_database = None
_gridfs = None


# ============================================================
# CONNECT TO MONGODB
# ============================================================


def connect_mongodb():
    """
    Connect to MongoDB.

    Reuses the existing MongoDB connection.
    """

    global _client
    global _database
    global _gridfs

    if not MONGO_URI:

        raise ValueError("MONGO_URI is missing. " "Configure it in the .env file.")

    try:

        if _client is None:

            client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
            )

            # Verify MongoDB connection
            client.admin.command("ping")

            # Assign connection after successful ping
            _client = client

            _database = client[MONGO_DATABASE]

            _gridfs = GridFS(
                _database,
                collection=MONGO_GRIDFS_BUCKET,
            )

            print(f"MongoDB connected successfully: " f"{MONGO_DATABASE}")

        return _database

    except PyMongoError as error:

        if _client is not None:
            _client.close()

        _client = None
        _database = None
        _gridfs = None

        raise ConnectionError(f"Could not connect to MongoDB: {error}") from error


# ============================================================
# GET DATABASE
# ============================================================


def get_database():
    """
    Return the MongoDB database.
    """

    if _database is None:

        return connect_mongodb()

    return _database


# ============================================================
# GET COLLECTION
# ============================================================


def get_collection(collection_name=None):
    """
    Return a MongoDB collection.

    Uses the configured prototype collection by default.
    """

    if collection_name is None:

        collection_name = MONGO_COLLECTION

    if not collection_name:

        raise ValueError("Collection name cannot be empty.")

    database = get_database()

    return database[collection_name]


# ============================================================
# GET GRIDFS
# ============================================================


def get_gridfs():
    """
    Return the GridFS instance.
    """

    global _gridfs

    if _gridfs is None:

        get_database()

    return _gridfs


# ============================================================
# GET GRIDFS FILE
# ============================================================


def get_gridfs_file(file_id):
    """
    Retrieve a GridFS file using its MongoDB file ID.
    """

    if file_id is None:

        raise ValueError("GridFS file ID cannot be empty.")

    gridfs_instance = get_gridfs()

    return gridfs_instance.get(file_id)


# ============================================================
# CLOSE MONGODB
# ============================================================


def close_mongodb():
    """
    Close the MongoDB connection.
    """

    global _client
    global _database
    global _gridfs

    if _client is not None:

        _client.close()

        _client = None
        _database = None
        _gridfs = None

        print("MongoDB connection closed.")
