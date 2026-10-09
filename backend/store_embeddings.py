"""
Generate and store jewellery catalogue embeddings in MongoDB.

- Uses the existing ONNX model and ModelLoader preprocessing.
- Reads images from GridFS.
- Stores 512-dimensional normalized embeddings in each catalogue document.
- Skips valid embeddings already stored for this model version.
- Does not modify or delete image files.
"""

from io import BytesIO
from pathlib import Path
import hashlib

import numpy as np
from bson import ObjectId
from gridfs import GridFS
from PIL import Image

from backend.database.mongodb import get_database
from backend.model_loader import ModelLoader

EMBEDDING_DIM = 512
MODEL_VERSION = "jewellery_dinov2_onnx_v1"
COLLECTIONS = ("gold", "prototype")


def get_gridfs_id(record):
    value = record.get("image_gridfs_id")
    if not value:
        return None

    try:
        return value if isinstance(value, ObjectId) else ObjectId(str(value))
    except Exception:
        return None


def is_valid_embedding(value):
    if not isinstance(value, list) or len(value) != EMBEDDING_DIM:
        return False

    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(vector).all():
        return False

    norm = float(np.linalg.norm(vector))
    return abs(norm - 1.0) < 1e-3


def main():
    print("=" * 70)
    print("JEWELMATCH AI - STORE CATALOGUE EMBEDDINGS")
    print("=" * 70)

    db = get_database()
    gridfs_store = GridFS(db)
    model = ModelLoader()

    # Fingerprint the actual ONNX file so changes to the model
    # can be distinguished from earlier embedding generations.
    from backend.model_loader import MODEL_PATH

    digest = hashlib.sha256()
    with open(MODEL_PATH, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    model_version = f"{MODEL_VERSION}:{digest.hexdigest()[:16]}"
    print(f"Model version: {model_version}")

    totals = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for collection_name in COLLECTIONS:
        collection = db[collection_name]

        print()
        print("-" * 70)
        print(f"COLLECTION: {collection_name}")
        print(f"Records: {collection.count_documents({})}")
        print("-" * 70)

        for record in collection.find({}):
            record_id = record.get("_id")
            totals["processed"] += 1

            # Skip only embeddings marked with this exact model
            # version and passing dimensional/numerical checks.
            if record.get("embedding_model") == model_version and is_valid_embedding(
                record.get("embedding")
            ):
                totals["skipped"] += 1
                print(f"[SKIP] {collection_name}/{record_id}: embedding exists")
                continue

            image = None

            try:
                gridfs_id = get_gridfs_id(record)
                if gridfs_id is None:
                    raise ValueError("Missing or invalid image_gridfs_id")

                if not gridfs_store.exists(gridfs_id):
                    raise FileNotFoundError(f"GridFS file not found: {gridfs_id}")

                image_bytes = gridfs_store.get(gridfs_id).read()
                if not image_bytes:
                    raise ValueError("GridFS image is empty")

                with Image.open(BytesIO(image_bytes)) as source:
                    image = source.convert("RGB")

                embedding = np.asarray(
                    model.get_embedding(image),
                    dtype=np.float32,
                ).reshape(-1)

                if embedding.size != EMBEDDING_DIM:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIM} values, "
                        f"received {embedding.size}"
                    )

                if not np.isfinite(embedding).all():
                    raise ValueError("Embedding contains non-finite values")

                norm = float(np.linalg.norm(embedding))
                if norm == 0.0:
                    raise ValueError("Embedding has zero norm")

                embedding = embedding / norm

                # Update only embedding-related fields in the existing
                # document. Catalogue metadata and image references stay
                # unchanged.
                result = collection.update_one(
                    {"_id": record_id},
                    {
                        "$set": {
                            "embedding": embedding.astype(np.float32).tolist(),
                            "embedding_model": model_version,
                        }
                    },
                )

                if result.matched_count != 1:
                    raise RuntimeError("MongoDB record disappeared during update")

                print(f"[OK] {collection_name}/{record_id}: saved 512 values")

            except Exception as exc:
                totals["failed"] += 1
                print(f"[ERROR] {collection_name}/{record_id}: {exc}")

            finally:
                if image is not None:
                    image.close()

    print()
    print("=" * 70)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 70)
    print(f"Records examined: {totals['processed']}")
    print(f"Already valid:    {totals['skipped']}")
    print(f"Failed:           {totals['failed']}")
    print(
        "Newly saved:      "
        f"{totals['processed'] - totals['skipped'] - totals['failed']}"
    )
    print("=" * 70)

    if totals["failed"]:
        raise SystemExit("Some records failed. Review the errors before deployment.")


if __name__ == "__main__":
    main()
