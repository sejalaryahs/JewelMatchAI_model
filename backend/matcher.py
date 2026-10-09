# ============================================================
# JEWELMATCH AI - MONGODB STORED-EMBEDDING MATCHER
# ============================================================

import gc

import numpy as np
from bson import ObjectId

from .database.mongodb import get_jewellery_collection
from .model_loader import ModelLoader


class JewelleryMatcher:
    """Match jewellery using embeddings stored in MongoDB."""

    EMBEDDING_DIM = 512
    MODEL_VERSION = "jewellery_dinov2_onnx_v1:cdae8a24b1a605da"

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.database = get_jewellery_collection().database

        self.collections = {
            "gold": self.database["gold"],
            "prototype": self.database["prototype"],
        }

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
        print("[INFO] Catalogue embeddings are read from MongoDB.")
        print("[INFO] Catalogue images are not re-embedded.")
        print("[INFO] Catalogue cache is cleared after each match.")
        print("=" * 70)

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

    def _get_item_id(self, item):
        for field in ("id", "design_id", "jewellery_id", "item_id"):
            value = item.get(field)
            if value is not None:
                return str(value)

        mongo_id = item.get("_id")
        return str(mongo_id) if mongo_id is not None else None

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

    def _create_embedding(self, image):
        """Generate an embedding for the uploaded query image."""
        embedding = self.model_loader.get_embedding(image)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.size != self.EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected embedding dimension: expected "
                f"{self.EMBEDDING_DIM}, received {embedding.size}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Query embedding contains NaN or infinite values.")

        norm = float(np.linalg.norm(embedding))

        if norm <= 0:
            raise ValueError("Query embedding has zero magnitude.")

        normalized = embedding / norm
        return normalized.astype(np.float32, copy=False)

    def _clear_catalogue_cache(self):
        """Remove references to cached catalogue records and vectors."""
        self.catalogue["gold"] = []
        self.catalogue["prototype"] = []

        self.embeddings["gold"] = None
        self.embeddings["prototype"] = None

        self._loaded_domain = None

    def _load_catalogue(self, domain):
        """Load validated precomputed embeddings from MongoDB."""
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

        for item in collection.find({}):
            total_records += 1

            if item.get("embedding_model") != self.MODEL_VERSION:
                wrong_model += 1
                continue

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

                vector = vector / norm
                gridfs_id = self._get_gridfs_id(item)

                if gridfs_id is None:
                    missing_image_id += 1

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

        print(f"[INFO] Target catalogue loaded: {domain}")
        print("=" * 70)

    def _ensure_catalogue_loaded(self, target):
        if target not in ("gold", "prototype"):
            raise ValueError("target must be 'gold' or 'prototype'")

        if self._loaded_domain == target and self.embeddings[target] is not None:
            return

        self._clear_catalogue_cache()
        self._load_catalogue(target)

    def find_matches(self, image, source, top_k=5):
        """Match one uploaded image and clear temporary cache afterwards."""
        query_embedding = None
        catalogue_embeddings = None
        catalogue_records = None
        similarities = None
        ranking = None
        results = []

        try:
            if source not in ("gold", "prototype"):
                raise ValueError("source must be 'gold' or 'prototype'")

            try:
                top_k = int(top_k)
            except (TypeError, ValueError):
                top_k = 5

            top_k = max(1, min(top_k, 50))
            target = "prototype" if source == "gold" else "gold"

            print()
            print("=" * 70)
            print("JEWELLERY MATCH")
            print("=" * 70)
            print(f"Source : {source}")
            print(f"Target : {target}")

            query_embedding = self._create_embedding(image)
            self._ensure_catalogue_loaded(target)

            catalogue_embeddings = self.embeddings[target]
            catalogue_records = self.catalogue[target]

            print(f"Target records: {len(catalogue_records)}")

            if not catalogue_records:
                print(f"[INFO] No usable {target} embeddings available.")
                return []

            # Preserve the existing cosine-similarity and ranking logic.
            similarities = catalogue_embeddings @ query_embedding
            ranking = np.argsort(similarities)[::-1][:top_k]

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
                    "image_gridfs_id": (
                        str(gridfs_id) if gridfs_id is not None else None
                    ),
                    "similarity": similarity,
                    "similarity_percent": round(
                        max(0.0, min(1.0, similarity)) * 100,
                        2,
                    ),
                }

                for field in (
                    "design_id",
                    "name",
                    "design_name",
                    "type",
                    "subtype",
                    "description",
                    "collection",
                ):
                    if field in item:
                        value = item[field]

                        if isinstance(value, ObjectId):
                            value = str(value)

                        result[field] = value

                results.append(result)

            print(f"Valid matches: {len(results)}")

            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. {result.get('id')} -> " f"{result['similarity_percent']}%"
                )

            print("=" * 70)

            # Return independent result dictionaries. They do not depend
            # on the catalogue cache after this method returns.
            return results

        finally:
            # Remove references to temporary NumPy arrays.
            query_embedding = None
            catalogue_embeddings = None
            catalogue_records = None
            similarities = None
            ranking = None

            # Clear both in-memory catalogue caches. MongoDB data is
            # untouched and will be loaded again for the next request.
            self._clear_catalogue_cache()

            # Ask Python to collect unreachable objects.
            gc.collect()

            print("[CLEANUP] Matching arrays and catalogue cache cleared.")
