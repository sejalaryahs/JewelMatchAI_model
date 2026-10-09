# ============================================================
# JEWELMATCH AI - MONGODB STORED-EMBEDDING MATCHER
#
# Preserves:
# - MongoDB gold and prototype collections
# - Existing trained ONNX ModelLoader
# - 512-dimensional embeddings
# - Cosine similarity and descending ranking
# - Existing matching result fields and metadata
#
# Optimized:
# - Loads precomputed embeddings from MongoDB
# - Does not generate embeddings for catalogue images
# - Does not download catalogue images from GridFS
# - Keeps only one target catalogue in memory
# ============================================================

import numpy as np
from bson import ObjectId

from .database.mongodb import get_jewellery_collection
from .model_loader import ModelLoader


class JewelleryMatcher:
    """Match jewellery using embeddings stored in MongoDB."""

    EMBEDDING_DIM = 512

    # Must match the version stored by backend/store_embeddings.py.
    MODEL_VERSION = "jewellery_dinov2_onnx_v1:cdae8a24b1a605da"

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

        # MongoDB database and collections.
        self.database = get_jewellery_collection().database

        self.collections = {
            "gold": self.database["gold"],
            "prototype": self.database["prototype"],
        }

        # One target catalogue is cached at a time.
        self.catalogue = {
            "gold": [],
            "prototype": [],
        }

        self.embeddings = {
            "gold": None,
            "prototype": None,
        }

        self._loaded_domain = None

        print()
        print("=" * 70)
        print("INITIALIZING JEWELMATCH STORED-EMBEDDING MATCHER")
        print("=" * 70)
        print(f"MongoDB database : {self.database.name}")
        print("Gold records     : " f"{self.collections['gold'].count_documents({})}")
        print(
            "Prototype records: " f"{self.collections['prototype'].count_documents({})}"
        )
        print(f"Embedding model  : {self.MODEL_VERSION}")
        print("[INFO] Catalogue embeddings will be read from MongoDB.")
        print("[INFO] Catalogue images will not be re-embedded.")
        print("[INFO] Only one target catalogue will be cached.")
        print("=" * 70)

    # ========================================================
    # GET GRIDFS ID
    # ========================================================

    def _get_gridfs_id(self, item):
        if not item:
            return None

        value = item.get("image_gridfs_id")

        if not value:
            return None

        try:
            if isinstance(value, ObjectId):
                return value

            return ObjectId(str(value))

        except Exception:
            return None

    # ========================================================
    # GET ITEM DISPLAY ID
    # ========================================================

    def _get_item_id(self, item):
        for field in (
            "id",
            "design_id",
            "jewellery_id",
            "item_id",
        ):
            value = item.get(field)

            if value is not None:
                return str(value)

        mongo_id = item.get("_id")

        if mongo_id is not None:
            return str(mongo_id)

        return None

    # ========================================================
    # GET IMAGE FILENAME
    # ========================================================

    def _get_filename(self, item):
        for field in (
            "filename",
            "image_filename",
            "original_filename",
            "image_name",
        ):
            value = item.get(field)

            if value:
                return str(value)

        return None

    # ========================================================
    # CREATE QUERY EMBEDDING
    # ========================================================

    def _create_embedding(self, image):
        """
        Generate an embedding only for the incoming query image.
        The existing trained ONNX model and preprocessing remain
        unchanged.
        """
        embedding = self.model_loader.get_embedding(image)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.size != self.EMBEDDING_DIM:
            raise ValueError(
                "Unexpected embedding dimension: "
                f"expected {self.EMBEDDING_DIM}, "
                f"received {embedding.size}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Query embedding contains NaN or infinite values.")

        norm = float(np.linalg.norm(embedding))

        if norm <= 0:
            raise ValueError("Query embedding has zero magnitude.")

        # Normalize so matrix multiplication computes cosine similarity.
        embedding = embedding / norm

        return embedding.astype(np.float32, copy=False)

    # ========================================================
    # CLEAR CATALOGUE CACHE
    # ========================================================

    def _clear_catalogue_cache(self):
        """Release references to the previous target catalogue."""
        self.catalogue["gold"] = []
        self.catalogue["prototype"] = []

        self.embeddings["gold"] = None
        self.embeddings["prototype"] = None

        self._loaded_domain = None

    # ========================================================
    # LOAD STORED EMBEDDINGS
    # ========================================================

    def _load_catalogue(self, domain):
        """
        Read precomputed embedding vectors from MongoDB.

        This method intentionally does not read GridFS image bytes
        and does not call the ONNX model for catalogue records.
        """
        if domain not in ("gold", "prototype"):
            raise ValueError("domain must be 'gold' or 'prototype'")

        print()
        print("=" * 70)
        print(f"LOADING STORED {domain.upper()} EMBEDDINGS")
        print("=" * 70)

        collection = self.collections[domain]

        valid_records = []
        vectors = []

        total_records = 0
        missing_embedding = 0
        wrong_model = 0
        invalid_embedding = 0
        missing_image_id = 0

        # Read one MongoDB document at a time.
        for item in collection.find({}):
            total_records += 1

            # Only accept vectors generated by the expected ONNX model.
            stored_model = item.get("embedding_model")

            if stored_model != self.MODEL_VERSION:
                wrong_model += 1
                continue

            # Each catalogue document must have a stored vector.
            stored_embedding = item.get("embedding")

            if stored_embedding is None:
                missing_embedding += 1
                continue

            try:
                vector = np.asarray(
                    stored_embedding,
                    dtype=np.float32,
                ).reshape(-1)

                if vector.size != self.EMBEDDING_DIM:
                    invalid_embedding += 1
                    continue

                if not np.all(np.isfinite(vector)):
                    invalid_embedding += 1
                    continue

                norm = float(np.linalg.norm(vector))

                if not np.isfinite(norm) or norm <= 0:
                    invalid_embedding += 1
                    continue

                # Stored vectors should already be normalized.
                # Normalize again to protect cosine-similarity ranking
                # against small floating-point differences.
                vector = vector / norm

                gridfs_id = self._get_gridfs_id(item)

                if gridfs_id is None:
                    missing_image_id += 1

                # Preserve the original MongoDB document and output
                # metadata without copying the image data.
                valid_records.append(
                    {
                        "record": item,
                        "gridfs_id": gridfs_id,
                    }
                )

                vectors.append(vector)

            except (TypeError, ValueError, OverflowError) as exc:
                invalid_embedding += 1
                print(
                    "[WARNING] Invalid stored embedding for "
                    f"{item.get('_id')}: {exc}"
                )

        if vectors:
            matrix = np.vstack(vectors).astype(
                np.float32,
                copy=False,
            )
        else:
            matrix = np.empty(
                (0, self.EMBEDDING_DIM),
                dtype=np.float32,
            )

        # Publish the completed cache.
        self.catalogue[domain] = valid_records
        self.embeddings[domain] = matrix
        self._loaded_domain = domain

        print(f"Total MongoDB records : {total_records}")
        print(f"Loaded embeddings     : {len(valid_records)}")
        print(f"Missing embeddings    : {missing_embedding}")
        print(f"Wrong model version   : {wrong_model}")
        print(f"Invalid embeddings    : {invalid_embedding}")
        print(f"Missing image IDs     : {missing_image_id}")
        print(f"Embedding matrix shape: {matrix.shape}")

        if not valid_records and total_records:
            print(
                "[WARNING] No usable stored embeddings were found. "
                "Check embedding_model and embedding fields."
            )

        print(f"[INFO] Cached target catalogue: {domain}")
        print("=" * 70)

    # ========================================================
    # ENSURE TARGET CATALOGUE IS LOADED
    # ========================================================

    def _ensure_catalogue_loaded(self, target):
        if target not in ("gold", "prototype"):
            raise ValueError("target must be 'gold' or 'prototype'")

        if self._loaded_domain == target and self.embeddings[target] is not None:
            return

        print(f"[INFO] Preparing target catalogue: {target}")

        # Free references to the previous catalogue before loading
        # the requested one.
        self._clear_catalogue_cache()
        self._load_catalogue(target)

    # ========================================================
    # MATCH
    # ========================================================

    def find_matches(self, image, source, top_k=5):
        if source not in ("gold", "prototype"):
            raise ValueError("source must be 'gold' or 'prototype'")

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

        top_k = max(1, min(top_k, 50))

        # Gold searches prototypes; prototypes search gold.
        target = "prototype" if source == "gold" else "gold"

        print()
        print("=" * 70)
        print("JEWELLERY MATCH")
        print("=" * 70)
        print(f"Source : {source}")
        print(f"Target : {target}")

        # Generate an embedding only for the uploaded image.
        query_embedding = self._create_embedding(image)

        # Read stored vectors for the target catalogue.
        self._ensure_catalogue_loaded(target)

        catalogue_embeddings = self.embeddings[target]
        catalogue_records = self.catalogue[target]

        print(f"Target records: {len(catalogue_records)}")

        if not catalogue_records:
            print(f"[INFO] No usable {target} embeddings available.")
            print("=" * 70)
            return []

        # Cosine similarity: normalized vectors multiplied together.
        similarities = catalogue_embeddings @ query_embedding

        # Rank from highest similarity to lowest.
        ranking = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in ranking:
            index = int(index)

            item_data = catalogue_records[index]
            item = item_data["record"]
            gridfs_id = item_data["gridfs_id"]

            similarity = float(similarities[index])

            result = {
                "id": self._get_item_id(item),
                "mongo_id": str(item.get("_id")),
                "filename": self._get_filename(item),
                "domain": target,
                "image_gridfs_id": (str(gridfs_id) if gridfs_id is not None else None),
                "similarity": similarity,
                "similarity_percent": round(
                    max(0.0, min(1.0, similarity)) * 100,
                    2,
                ),
            }

            # Preserve the existing metadata fields.
            metadata_fields = (
                "design_id",
                "name",
                "design_name",
                "type",
                "subtype",
                "description",
                "collection",
            )

            for field in metadata_fields:
                if field in item:
                    value = item[field]

                    if isinstance(value, ObjectId):
                        value = str(value)

                    result[field] = value

            results.append(result)

        print()
        print(f"Valid matches: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result.get('id')} -> " f"{result['similarity_percent']}%")

        print("=" * 70)

        return results
