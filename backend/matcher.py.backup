# ============================================================
# JEWELMATCH AI - MONGODB / GRIDFS DINO JEWELLERY MATCHER
# ============================================================

from io import BytesIO

import numpy as np
from bson import ObjectId
from gridfs import GridFS
from PIL import Image

from .database.mongodb import get_jewellery_collection
from .model_loader import ModelLoader

# ============================================================
# JEWELLERY MATCHER
# ============================================================


class JewelleryMatcher:

    def __init__(
        self,
        model_loader: ModelLoader,
    ):

        self.model_loader = model_loader

        # ----------------------------------------------------
        # EXISTING MONGODB DATABASE
        # ----------------------------------------------------

        self.database = get_jewellery_collection().database

        # ----------------------------------------------------
        # GOLD / PROTOTYPE COLLECTIONS
        # ----------------------------------------------------

        self.collections = {
            "gold": self.database["gold"],
            "prototype": self.database["prototype"],
        }

        # ----------------------------------------------------
        # GRIDFS
        # ----------------------------------------------------

        self.gridfs = GridFS(self.database)

        # ----------------------------------------------------
        # CATALOGUE
        #
        # Same concept as the old localhost matcher:
        #
        # self.catalogue["gold"]
        # self.catalogue["prototype"]
        #
        # But instead of storing local paths, we store the
        # MongoDB record and GridFS image information.
        # ----------------------------------------------------

        self.catalogue = {
            "gold": [],
            "prototype": [],
        }

        # ----------------------------------------------------
        # EMBEDDINGS
        #
        # Same concept as the old localhost matcher:
        #
        # self.embeddings["gold"]
        # self.embeddings["prototype"]
        #
        # Embeddings are created ONCE during initialization.
        # ----------------------------------------------------

        self.embeddings = {
            "gold": None,
            "prototype": None,
        }

        print()
        print("=" * 70)
        print("INITIALIZING JEWELLERY MONGODB MATCHER")
        print("=" * 70)

        print(f"MongoDB database : {self.database.name}")

        print(f"Gold records     : " f"{self.collections['gold'].count_documents({})}")

        print(
            f"Prototype records: "
            f"{self.collections['prototype'].count_documents({})}"
        )

        # ----------------------------------------------------
        # LOAD MONGODB CATALOGUE
        # ----------------------------------------------------

        self._load_catalogue()

        print("=" * 70)

    # ========================================================
    # GET GRIDFS ID
    # ========================================================

    def _get_gridfs_id(
        self,
        item,
    ):

        if not item:
            return None

        value = item.get("image_gridfs_id")

        if not value:
            return None

        try:

            if isinstance(
                value,
                ObjectId,
            ):

                return value

            return ObjectId(str(value))

        except Exception:

            return None

    # ========================================================
    # GET GRIDFS IMAGE
    # ========================================================

    def _get_gridfs_image(
        self,
        gridfs_id,
    ):

        if not gridfs_id:
            return None

        try:

            # ------------------------------------------------
            # CHECK FILE
            # ------------------------------------------------

            if not self.gridfs.exists(gridfs_id):

                print("[WARNING] GridFS image " f"not found: {gridfs_id}")

                return None

            # ------------------------------------------------
            # READ GRIDFS FILE
            # ------------------------------------------------

            grid_file = self.gridfs.get(gridfs_id)

            image_bytes = grid_file.read()

            if not image_bytes:

                print("[WARNING] Empty GridFS " f"image: {gridfs_id}")

                return None

            # ------------------------------------------------
            # CONVERT TO PIL
            # ------------------------------------------------

            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            return image

        except Exception as exc:

            print("[WARNING] Could not read " f"GridFS image {gridfs_id}: " f"{exc}")

            return None

    # ========================================================
    # GET ITEM DISPLAY ID
    # ========================================================

    def _get_item_id(
        self,
        item,
    ):

        # ----------------------------------------------------
        # Prefer application-level ID
        # ----------------------------------------------------

        for field in (
            "id",
            "design_id",
            "jewellery_id",
            "item_id",
        ):

            value = item.get(field)

            if value is not None:

                return str(value)

        # ----------------------------------------------------
        # MongoDB ID fallback
        # ----------------------------------------------------

        mongo_id = item.get("_id")

        if mongo_id is not None:

            return str(mongo_id)

        return None

    # ========================================================
    # GET IMAGE FILENAME
    # ========================================================

    def _get_filename(
        self,
        item,
    ):

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

    def _create_embedding(
        self,
        image,
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use exactly the same ModelLoader used by the old
        # localhost matcher.
        #
        # ModelLoader already normalizes the embedding.
        # ----------------------------------------------------

        embedding = self.model_loader.get_embedding(image)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        return embedding

    # ========================================================
    # LOAD CATALOGUE
    #
    # This replaces the old local:
    #
    # matching_data/gold
    # matching_data/prototypes
    #
    # with:
    #
    # MongoDB gold
    # MongoDB prototype
    # GridFS images
    # ========================================================

    def _load_catalogue(
        self,
    ):

        print()
        print("=" * 70)
        print("LOADING JEWELLERY MONGODB CATALOGUE")
        print("=" * 70)

        for domain in (
            "gold",
            "prototype",
        ):

            collection = self.collections[domain]

            records = list(collection.find({}))

            print()
            print(f"Loading {domain} catalogue...")

            valid_records = []
            vectors = []

            for item in records:

                try:

                    # ----------------------------------------
                    # GET GRIDFS IMAGE ID
                    # ----------------------------------------

                    gridfs_id = self._get_gridfs_id(item)

                    if not gridfs_id:

                        print(
                            "[WARNING] Skipping "
                            f"{domain} record "
                            f"{item.get('_id')} "
                            "without image_gridfs_id"
                        )

                        continue

                    # ----------------------------------------
                    # GET IMAGE
                    # ----------------------------------------

                    image = self._get_gridfs_image(gridfs_id)

                    if image is None:

                        continue

                    # ----------------------------------------
                    # CREATE DINO EMBEDDING
                    # ----------------------------------------

                    embedding = self._create_embedding(image)

                    # ----------------------------------------
                    # STORE RECORD
                    # ----------------------------------------

                    valid_records.append(
                        {
                            "record": item,
                            "gridfs_id": gridfs_id,
                        }
                    )

                    vectors.append(embedding)

                except Exception as exc:

                    print(
                        "[WARNING] Could not "
                        f"process {domain} record "
                        f"{item.get('_id')}: "
                        f"{exc}"
                    )

                    continue

            # ------------------------------------------------
            # STORE CATALOGUE
            # ------------------------------------------------

            self.catalogue[domain] = valid_records

            # ------------------------------------------------
            # STORE EMBEDDING MATRIX
            # ------------------------------------------------

            if not vectors:

                self.embeddings[domain] = np.empty(
                    (
                        0,
                        512,
                    ),
                    dtype=np.float32,
                )

                print(f"{domain.capitalize()} embeddings: 0")

                continue

            matrix = np.vstack(vectors).astype(np.float32)

            self.embeddings[domain] = matrix

            print(f"{domain.capitalize()} images      : " f"{len(valid_records)}")

            print(f"{domain.capitalize()} embeddings  : " f"{len(matrix)}")

        print()
        print("=" * 70)
        print("MONGODB CATALOGUE READY")
        print("=" * 70)

    # ========================================================
    # MATCH
    #
    # This intentionally follows the old localhost matcher:
    #
    # query embedding
    #       ↓
    # catalogue embedding matrix
    #       ↓
    # matrix multiplication
    #       ↓
    # argsort
    #       ↓
    # top K
    # ========================================================

    def find_matches(
        self,
        image,
        source,
        top_k=5,
    ):

        # ----------------------------------------------------
        # VALIDATE SOURCE
        # ----------------------------------------------------

        if source not in {
            "gold",
            "prototype",
        }:

            raise ValueError("source must be 'gold' " "or 'prototype'")

        # ----------------------------------------------------
        # VALIDATE TOP K
        # ----------------------------------------------------

        try:

            top_k = int(top_k)

        except Exception:

            top_k = 5

        top_k = max(
            1,
            min(
                top_k,
                50,
            ),
        )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target = "prototype" if source == "gold" else "gold"

        print()
        print("=" * 70)
        print("JEWELLERY MATCH")
        print("=" * 70)

        print(f"Source : {source}")

        print(f"Target : {target}")

        # ----------------------------------------------------
        # CREATE QUERY EMBEDDING
        #
        # Same as old localhost matcher.
        # ----------------------------------------------------

        query_embedding = self._create_embedding(image)

        # ----------------------------------------------------
        # GET PRECOMPUTED TARGET EMBEDDINGS
        # ----------------------------------------------------

        catalogue_embeddings = self.embeddings[target]

        catalogue_records = self.catalogue[target]

        print(f"Target records: " f"{len(catalogue_records)}")

        # ----------------------------------------------------
        # EMPTY CATALOGUE
        # ----------------------------------------------------

        if len(catalogue_records) == 0:

            print(f"[INFO] No {target} " "records available.")

            print("=" * 70)

            return []

        # ----------------------------------------------------
        # NORMALIZE QUERY
        #
        # ModelLoader already does this, but this keeps the
        # matrix multiplication mathematically safe.
        # ----------------------------------------------------

        query_norm = np.linalg.norm(query_embedding)

        if query_norm > 0:

            query_embedding = query_embedding / query_norm

        # ----------------------------------------------------
        # COSINE SIMILARITY
        #
        # EXACT SAME OPERATION AS OLD LOCAL MATCHER:
        #
        # similarities =
        #     catalogue_embeddings @ query_embedding
        # ----------------------------------------------------

        similarities = catalogue_embeddings @ query_embedding

        # ----------------------------------------------------
        # SORT DESCENDING
        #
        # EXACT SAME OPERATION AS OLD LOCAL MATCHER.
        # ----------------------------------------------------

        ranking = np.argsort(similarities)[::-1]

        ranking = ranking[:top_k]

        # ----------------------------------------------------
        # BUILD RESULTS
        # ----------------------------------------------------

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
                "filename": (self._get_filename(item)),
                "domain": target,
                "image_gridfs_id": str(gridfs_id),
                "similarity": similarity,
                "similarity_percent": round(
                    max(
                        0.0,
                        min(
                            1.0,
                            similarity,
                        ),
                    )
                    * 100,
                    2,
                ),
            }

            # ------------------------------------------------
            # ADD AVAILABLE METADATA
            # ------------------------------------------------

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

                    if isinstance(
                        value,
                        ObjectId,
                    ):

                        value = str(value)

                    result[field] = value

            results.append(result)

        # ----------------------------------------------------
        # LOG RESULTS
        # ----------------------------------------------------

        print()

        print(f"Valid matches: " f"{len(results)}")

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"{rank}. "
                f"{result.get('id')} "
                f"-> "
                f"{result['similarity_percent']}%"
            )

        print("=" * 70)

        return results
