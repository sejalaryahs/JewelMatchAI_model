# ============================================================
# JEWELMATCH AI - MONGODB / GRIDFS DINO JEWELLERY MATCHER
# Lazy catalogue loading to reduce startup memory pressure.
#
# Preserves:
# - MongoDB and GridFS
# - Existing ONNX ModelLoader
# - 512-dimensional embeddings
# - Cosine similarity and descending ranking
# - Existing matching result fields
# ============================================================

from io import BytesIO

import numpy as np
from bson import ObjectId
from gridfs import GridFS
from PIL import Image

from .database.mongodb import get_jewellery_collection
from .model_loader import ModelLoader


class JewelleryMatcher:
    """
    Jewellery matcher with lazy, single-catalogue embedding cache.

    The ONNX model is loaded by app.py. Catalogue embeddings
    are generated only when a matching request needs them.

    Only one target catalogue is retained in the cache at a time.
    """

    EMBEDDING_DIM = 512

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

        # MongoDB database and collections.
        self.database = get_jewellery_collection().database

        self.collections = {
            "gold": self.database["gold"],
            "prototype": self.database["prototype"],
        }

        self.gridfs = GridFS(self.database)

        # Keep the existing catalogue structure.
        self.catalogue = {
            "gold": [],
            "prototype": [],
        }

        self.embeddings = {
            "gold": None,
            "prototype": None,
        }

        # Identifies the one catalogue currently cached.
        self._loaded_domain = None

        print()
        print("=" * 70)
        print("INITIALIZING JEWELLERY MONGODB MATCHER")
        print("=" * 70)

        print(f"MongoDB database : {self.database.name}")
        print("Gold records     : " f"{self.collections['gold'].count_documents({})}")
        print(
            "Prototype records: " f"{self.collections['prototype'].count_documents({})}"
        )

        # IMPORTANT:
        # Do not generate catalogue embeddings during startup.
        print("[INFO] Catalogue embeddings will load on demand.")
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
    # GET GRIDFS IMAGE
    # ========================================================

    def _get_gridfs_image(self, gridfs_id):
        if not gridfs_id:
            return None

        try:
            if not self.gridfs.exists(gridfs_id):
                print("[WARNING] GridFS image not found: " f"{gridfs_id}")
                return None

            grid_file = self.gridfs.get(gridfs_id)
            image_bytes = grid_file.read()

            if not image_bytes:
                print("[WARNING] Empty GridFS image: " f"{gridfs_id}")
                return None

            with Image.open(BytesIO(image_bytes)) as source_image:
                # Return an independent image after the byte stream
                # and original PIL image have been closed.
                image = source_image.convert("RGB")

            return image

        except Exception as exc:
            print("[WARNING] Could not read GridFS image " f"{gridfs_id}: {exc}")
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
    # CREATE EMBEDDING
    # ========================================================

    def _create_embedding(self, image):
        # ModelLoader uses the existing trained ONNX model and
        # performs the existing image preprocessing.
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

        return embedding

    # ========================================================
    # CLEAR CATALOGUE CACHE
    # ========================================================

    def _clear_catalogue_cache(self):
        """
        Release references to the previous catalogue before
        building another one.

        This does not force Python or ONNX Runtime to return
        all allocated memory to the operating system.
        """
        self.catalogue["gold"] = []
        self.catalogue["prototype"] = []

        self.embeddings["gold"] = None
        self.embeddings["prototype"] = None

        self._loaded_domain = None

    # ========================================================
    # LOAD ONE CATALOGUE
    # ========================================================

    def _load_catalogue(self, domain):
        """
        Build embeddings for only the requested catalogue.

        The existing matching algorithm is unchanged. This
        method changes when catalogue embeddings are generated,
        not how similarity is calculated.
        """
        if domain not in ("gold", "prototype"):
            raise ValueError("domain must be 'gold' or 'prototype'")

        print()
        print("=" * 70)
        print(f"LOADING {domain.upper()} CATALOGUE")
        print("=" * 70)

        collection = self.collections[domain]

        valid_records = []
        vectors = []

        # Iterate the MongoDB cursor instead of materializing
        # every database document in a separate list.
        for item in collection.find({}):
            image = None

            try:
                gridfs_id = self._get_gridfs_id(item)

                if not gridfs_id:
                    print(
                        "[WARNING] Skipping "
                        f"{domain} record {item.get('_id')} "
                        "without image_gridfs_id"
                    )
                    continue

                image = self._get_gridfs_image(gridfs_id)

                if image is None:
                    continue

                embedding = self._create_embedding(image)

                valid_records.append(
                    {
                        "record": item,
                        "gridfs_id": gridfs_id,
                    }
                )

                vectors.append(embedding)

            except Exception as exc:
                print(
                    "[WARNING] Could not process "
                    f"{domain} record {item.get('_id')}: "
                    f"{exc}"
                )

            finally:
                # Do not retain decoded catalogue images.
                if image is not None:
                    image.close()

        # Build only one embedding matrix.
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

        # Publish the completed catalogue cache.
        self.catalogue[domain] = valid_records
        self.embeddings[domain] = matrix
        self._loaded_domain = domain

        print(f"{domain.capitalize()} images     : {len(valid_records)}")
        print(f"{domain.capitalize()} embeddings : {len(matrix)}")
        print(f"[INFO] Cached target catalogue: {domain}")
        print("=" * 70)

    # ========================================================
    # ENSURE TARGET CATALOGUE IS LOADED
    # ========================================================

    def _ensure_catalogue_loaded(self, target):
        """
        Reuse the current cache when it matches the requested
        target. Otherwise clear it before loading the new target.
        """
        if self._loaded_domain == target:
            if self.embeddings[target] is not None:
                return

        print(f"[INFO] Preparing target catalogue: {target}")

        # Clear the previous cache BEFORE loading the next one.
        self._clear_catalogue_cache()

        self._load_catalogue(target)

    # ========================================================
    # MATCH
    # ========================================================

    def find_matches(self, image, source, top_k=5):
        # Validate source.
        if source not in ("gold", "prototype"):
            raise ValueError("source must be 'gold' or 'prototype'")

        # Validate top_k using the existing limits.
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

        # Create the uploaded/query image embedding.
        query_embedding = self._create_embedding(image)

        # Load only the target catalogue when necessary.
        self._ensure_catalogue_loaded(target)

        catalogue_embeddings = self.embeddings[target]
        catalogue_records = self.catalogue[target]

        print(f"Target records: {len(catalogue_records)}")

        if not catalogue_records:
            print(f"[INFO] No {target} records available.")
            print("=" * 70)
            return []

        # Keep the original query normalization.
        query_norm = np.linalg.norm(query_embedding)

        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # ORIGINAL MATCHING ALGORITHM:
        # cosine similarity using matrix multiplication.
        similarities = catalogue_embeddings @ query_embedding

        # ORIGINAL RANKING:
        # descending similarity, then select top_k.
        ranking = np.argsort(similarities)[::-1]
        ranking = ranking[:top_k]

        # Build the existing result structure.
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
                "image_gridfs_id": str(gridfs_id),
                "similarity": similarity,
                "similarity_percent": round(
                    max(0.0, min(1.0, similarity)) * 100,
                    2,
                ),
            }

            # Preserve available metadata fields.
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
