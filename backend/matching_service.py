import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from training.augmentations import get_validation_transform

from .database.supabase import supabase
from .model_loader import ModelLoader
from .matcher import JewelleryMatcher
from .segmentation import segment_jewellery

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_TABLE = "jewellery_embeddings_v3"
JEWELLERY_TABLE = "jewellery"

# MUST MATCH THE VALUE STORED IN SUPABASE.
EMBEDDING_MODEL = "jewellery_model3_onnx_v1"

EMBEDDING_DIM = 512
TOP_K = 12

GOLD_TYPE = "gold"
PROTOTYPE_TYPE = "prototype"

COLLECTION_NAMES = {
    GOLD_TYPE: "Gold",
    PROTOTYPE_TYPE: "Prototype",
}

PAGE_SIZE = 100


# ============================================================
# MATCHING SERVICE
# ============================================================


class JewelleryMatchingService:

    def __init__(self):

        print()
        print("=" * 70)
        print("JEWELLERY MODEL 3 - BIDIRECTIONAL MATCHING SERVICE")
        print("=" * 70)

        # ONNX model
        self.model_loader = ModelLoader()
        self.model_loader.load_model()

        # Validation preprocessing
        self.transform = get_validation_transform()

        # Matcher
        self.matcher = JewelleryMatcher(
            embedding_dim=EMBEDDING_DIM,
            top_k=TOP_K,
            similarity_threshold=0.0,
        )

        self.supabase = supabase

        print()
        print("Model loaded successfully.")
        print("Runtime: ONNX Runtime")
        print("Embedding model:", EMBEDDING_MODEL)
        print("Embedding dimension:", EMBEDDING_DIM)
        print("Number of results:", TOP_K)
        print("Embedding table:", EMBEDDING_TABLE)
        print("Jewellery table:", JEWELLERY_TABLE)

        print()
        print("Supported matching directions:")
        print("1. Gold -> Prototype")
        print("2. Prototype -> Gold")
        print("=" * 70)

    # ========================================================
    # GENERATE QUERY EMBEDDING USING ONNX
    # ========================================================

    def generate_embedding(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with tempfile.TemporaryDirectory() as temp_dir:

            segmented_path = Path(temp_dir) / "segmented_image.png"

            segment_jewellery(
                input_path=image_path,
                output_path=segmented_path,
            )

            if not segmented_path.exists():
                raise RuntimeError("Jewellery segmentation did not produce an image.")

            with Image.open(segmented_path) as image:

                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                image_tensor = self.transform(image)

            # Convert transformed image to ONNX input.
            image_array = image_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)

            expected_shape = (1, 3, 224, 224)

            if image_array.shape != expected_shape:
                raise ValueError(
                    f"Expected ONNX input {expected_shape}, "
                    f"received {image_array.shape}"
                )

            # ONNX inference.
            embedding = self.model_loader.get_embeddings(image_array)

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            if embedding.shape != (1, EMBEDDING_DIM):
                raise ValueError(
                    f"Expected embedding shape (1, {EMBEDDING_DIM}), "
                    f"received {embedding.shape}"
                )

            if not np.isfinite(embedding).all():
                raise ValueError("Generated embedding contains invalid values.")

            # L2 normalization.
            norm = np.linalg.norm(
                embedding,
                axis=1,
                keepdims=True,
            )

            if np.any(norm <= 1e-12):
                raise ValueError("Generated embedding has zero magnitude.")

            embedding = embedding / norm

            return embedding[0].astype(np.float32)

    # ========================================================
    # PARSE SUPABASE EMBEDDING
    # ========================================================

    @staticmethod
    def parse_embedding(value):

        if value is None:
            return None

        if isinstance(value, str):

            value = value.strip()

            try:
                value = json.loads(value)

            except json.JSONDecodeError:

                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1]

                value = np.fromstring(
                    value,
                    sep=",",
                    dtype=np.float32,
                )

        try:

            embedding = np.asarray(
                value,
                dtype=np.float32,
            ).reshape(-1)

        except (TypeError, ValueError):
            return None

        if embedding.shape != (EMBEDDING_DIM,):
            return None

        if not np.isfinite(embedding).all():
            return None

        norm = np.linalg.norm(embedding)

        if norm <= 1e-12:
            return None

        return (embedding / norm).astype(np.float32)

    # ========================================================
    # FETCH ALL EMBEDDING RECORDS
    # ========================================================

    def fetch_embedding_records(self):

        records = []
        start = 0

        while True:

            end = start + PAGE_SIZE - 1

            response = (
                self.supabase.table(EMBEDDING_TABLE)
                .select(
                    "id, jewellery_id, embedding_model, "
                    "embedding_dimension, embedding"
                )
                .eq(
                    "embedding_model",
                    EMBEDDING_MODEL,
                )
                .eq(
                    "embedding_dimension",
                    EMBEDDING_DIM,
                )
                .range(start, end)
                .execute()
            )

            page = response.data or []

            if not page:
                break

            records.extend(page)

            if len(page) < PAGE_SIZE:
                break

            start += PAGE_SIZE

        return records

    # ========================================================
    # FETCH JEWELLERY DETAILS
    # ========================================================

    def fetch_jewellery_details(self, jewellery_ids):

        jewellery_lookup = {}

        # Keep ID batches manageable.
        batch_size = 100

        for start in range(0, len(jewellery_ids), batch_size):

            batch_ids = jewellery_ids[start : start + batch_size]

            response = (
                self.supabase.table(JEWELLERY_TABLE)
                .select(
                    "id, name, collection, type, description, "
                    "image_path, image_url, design_code"
                )
                .in_(
                    "id",
                    batch_ids,
                )
                .execute()
            )

            for item in response.data or []:

                if item.get("id"):
                    jewellery_lookup[str(item["id"])] = item

        return jewellery_lookup

    # ========================================================
    # LOAD TARGET EMBEDDINGS
    # ========================================================

    def get_target_embeddings(self, target_type):

        target_type = str(target_type).strip().lower()

        if target_type not in (
            GOLD_TYPE,
            PROTOTYPE_TYPE,
        ):
            raise ValueError("Expected target type 'gold' or 'prototype'.")

        target_collection = COLLECTION_NAMES[target_type]

        print()
        print("=" * 70)
        print(f"LOADING {target_collection.upper()} EMBEDDINGS")
        print("=" * 70)

        print("Expected embedding model:", EMBEDDING_MODEL)
        print("Expected dimension:", EMBEDDING_DIM)

        # Retrieve embeddings with pagination.
        embedding_records = self.fetch_embedding_records()

        print(
            "Total matching-model embedding records:",
            len(embedding_records),
        )

        if not embedding_records:

            print(
                "No embeddings found for model:",
                EMBEDDING_MODEL,
            )

            return []

        jewellery_ids = list(
            dict.fromkeys(
                str(record["jewellery_id"])
                for record in embedding_records
                if record.get("jewellery_id")
            )
        )

        if not jewellery_ids:

            print("No jewellery IDs found.")
            return []

        jewellery_lookup = self.fetch_jewellery_details(jewellery_ids)

        target_items = []

        for record in embedding_records:

            jewellery_id = record.get("jewellery_id")

            if not jewellery_id:
                continue

            jewellery = jewellery_lookup.get(str(jewellery_id))

            if not jewellery:
                continue

            # IMPORTANT:
            # Your catalogue stores Gold/Prototype in collection.
            collection = str(jewellery.get("collection") or "").strip().lower()

            if collection != target_collection.lower():
                continue

            embedding = self.parse_embedding(record.get("embedding"))

            if embedding is None:

                print(
                    "Skipping invalid embedding:",
                    jewellery_id,
                )

                continue

            target_items.append(
                {
                    "id": str(jewellery["id"]),
                    "embedding": embedding,
                    "embedding_model": record.get("embedding_model"),
                    "embedding_dimension": record.get("embedding_dimension"),
                    "name": jewellery.get("name"),
                    "design_code": jewellery.get("design_code"),
                    "collection": jewellery.get("collection"),
                    "type": jewellery.get("type"),
                    "description": jewellery.get("description"),
                    "image_path": jewellery.get("image_path"),
                    "image_url": jewellery.get("image_url"),
                }
            )

        print(
            f"Valid {target_collection} embeddings:",
            len(target_items),
        )

        return target_items

    # ========================================================
    # GET GOLD EMBEDDINGS
    # ========================================================

    def get_gold_embeddings(self):

        return self.get_target_embeddings(GOLD_TYPE)

    # ========================================================
    # GET PROTOTYPE EMBEDDINGS
    # ========================================================

    def get_prototype_embeddings(self):

        return self.get_target_embeddings(PROTOTYPE_TYPE)

    # ========================================================
    # MATCH IMAGE
    # ========================================================

    def match_image(
        self,
        image_path,
        query_type,
        top_k=TOP_K,
    ):

        query_type = str(query_type).strip().lower()

        if query_type not in (
            GOLD_TYPE,
            PROTOTYPE_TYPE,
        ):
            raise ValueError("query_type must be 'gold' or 'prototype'.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Select opposite category.
        target_type = PROTOTYPE_TYPE if query_type == GOLD_TYPE else GOLD_TYPE

        print()
        print("=" * 70)
        print("JEWELLERY IMAGE MATCHING")
        print("=" * 70)

        print("Uploaded image type:", query_type)
        print("Searching for:", target_type)
        print("Requested results:", top_k)

        # Generate query embedding.
        query_embedding = self.generate_embedding(image_path)

        print(
            "Query embedding shape:",
            query_embedding.shape,
        )

        # Fetch target catalogue embeddings.
        target_items = self.get_target_embeddings(target_type)

        if not target_items:

            print(f"No valid {target_type} embeddings found.")

            return []

        print(
            "Available target designs:",
            len(target_items),
        )

        # The matcher can return no more than the available
        # valid target records.
        actual_top_k = min(
            top_k,
            len(target_items),
        )

        results = self.matcher.match(
            query_embedding=query_embedding,
            prototypes=target_items,
            top_k=actual_top_k,
        )

        matching_results = []

        for result in results:

            matching_results.append(self.matcher.get_match_info(result))

        print()
        print("Matching completed.")
        print("Uploaded type:", query_type)
        print("Target type:", target_type)
        print("Results returned:", len(matching_results))
        print("=" * 70)

        return matching_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    service = JewelleryMatchingService()

    gold_items = service.get_gold_embeddings()

    prototype_items = service.get_prototype_embeddings()

    print()
    print("=" * 70)
    print("SUPABASE EMBEDDING SUMMARY")
    print("=" * 70)

    print("Gold embeddings:", len(gold_items))
    print("Prototype embeddings:", len(prototype_items))
    print("Requested matching results:", TOP_K)

    print("=" * 70)
