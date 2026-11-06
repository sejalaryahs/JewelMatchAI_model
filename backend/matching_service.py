import tempfile
from pathlib import Path

import torch

from training.augmentations import get_validation_transform

from .config import DEVICE
from .database.supabase import supabase
from .model_loader import ModelLoader
from .matcher import JewelleryMatcher
from .segmentation import segment_jewellery

# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_TABLE = "jewellery_embeddings_v3"

JEWELLERY_TABLE = "jewellery"

EMBEDDING_MODEL = "jewellery_dinov2_v3"

EMBEDDING_DIM = 512

TOP_K = 5


# ============================================================
# MATCHING SERVICE
# ============================================================


class JewelleryMatchingService:

    def __init__(self):

        print()
        print("=" * 70)
        print("JEWELLERY MODEL 3 MATCHING SERVICE")
        print("=" * 70)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model_loader = ModelLoader()

        self.model = self.model_loader.load_model()

        self.device = self.model_loader.get_device()

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        self.transform = get_validation_transform()

        # ----------------------------------------------------
        # MATCHER
        # ----------------------------------------------------

        self.matcher = JewelleryMatcher(
            embedding_dim=EMBEDDING_DIM,
            top_k=TOP_K,
            similarity_threshold=0.0,
        )

        # ----------------------------------------------------
        # SUPABASE
        # ----------------------------------------------------

        self.supabase = supabase

        print()
        print("Model loaded successfully.")
        print("Device:", self.device)
        print("Embedding dimension:", EMBEDDING_DIM)
        print("Embedding table:", EMBEDDING_TABLE)
        print("Jewellery table:", JEWELLERY_TABLE)

        print("=" * 70)

    # ========================================================
    # GENERATE QUERY EMBEDDING
    # ========================================================

    def generate_embedding(
        self,
        image_path: str | Path,
    ):

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(f"Image not found: {image_path}")

        # ----------------------------------------------------
        # SEGMENT IMAGE
        # ----------------------------------------------------

        with tempfile.TemporaryDirectory() as temp_dir:

            segmented_path = Path(temp_dir) / "segmented_image.jpg"

            segment_jewellery(
                input_path=image_path,
                output_path=segmented_path,
            )

            # ------------------------------------------------
            # LOAD IMAGE
            # ------------------------------------------------

            from PIL import Image

            with Image.open(segmented_path) as image:

                image = image.convert("RGB")

                image_tensor = self.transform(image)

            # ------------------------------------------------
            # MODEL INPUT
            # ------------------------------------------------

            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            self.model.eval()

            with torch.no_grad():

                embedding = self.model(image_tensor)

            # ------------------------------------------------
            # VALIDATE OUTPUT
            # ------------------------------------------------

            if embedding.ndim != 2:

                raise ValueError(
                    "Expected model output shape "
                    "(1, 512), received "
                    f"{tuple(embedding.shape)}"
                )

            if embedding.shape[1] != EMBEDDING_DIM:

                raise ValueError(
                    f"Expected {EMBEDDING_DIM} dimensions, "
                    f"received {embedding.shape[1]}"
                )

            if not torch.isfinite(embedding).all():

                raise ValueError("Generated embedding contains invalid values.")

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1,
            )

            return embedding.squeeze(0).cpu().numpy()

    # ========================================================
    # LOAD PROTOTYPE EMBEDDINGS FROM SUPABASE
    # ========================================================

    def get_prototype_embeddings(self):

        print()
        print("Loading prototype embeddings from Supabase...")

        # ----------------------------------------------------
        # RETRIEVE EMBEDDINGS
        # ----------------------------------------------------

        response = (
            self.supabase.table(EMBEDDING_TABLE)
            .select(
                "id, jewellery_id, embedding_model, " "embedding_dimension, embedding"
            )
            .eq(
                "embedding_model",
                EMBEDDING_MODEL,
            )
            .eq(
                "embedding_dimension",
                EMBEDDING_DIM,
            )
            .execute()
        )

        embedding_records = response.data or []

        print(
            "Supabase embedding records:",
            len(embedding_records),
        )

        if not embedding_records:

            print("No embeddings found in Supabase.")

            return []

        # ----------------------------------------------------
        # GET JEWELLERY IDS
        # ----------------------------------------------------

        jewellery_ids = list(
            dict.fromkeys(
                record["jewellery_id"]
                for record in embedding_records
                if record.get("jewellery_id")
            )
        )

        if not jewellery_ids:

            print("No jewellery IDs found.")

            return []

        # ----------------------------------------------------
        # RETRIEVE JEWELLERY DETAILS
        # ----------------------------------------------------

        jewellery_response = (
            self.supabase.table(JEWELLERY_TABLE)
            .select(
                "id, name, collection, type, description, "
                "image_path, image_url, design_code"
            )
            .in_(
                "id",
                jewellery_ids,
            )
            .execute()
        )

        jewellery_records = jewellery_response.data or []

        print(
            "Supabase jewellery records:",
            len(jewellery_records),
        )

        # ----------------------------------------------------
        # CREATE JEWELLERY LOOKUP
        # ----------------------------------------------------

        jewellery_lookup = {
            str(item["id"]): item for item in jewellery_records if item.get("id")
        }

        # ----------------------------------------------------
        # COMBINE EMBEDDINGS AND JEWELLERY DETAILS
        # ----------------------------------------------------

        prototypes = []

        for record in embedding_records:

            jewellery_id = record.get("jewellery_id")

            if not jewellery_id:

                continue

            jewellery = jewellery_lookup.get(str(jewellery_id))

            if not jewellery:

                print(
                    "Jewellery record not found:",
                    jewellery_id,
                )

                continue

            embedding = record.get("embedding")

            if embedding is None:

                continue

            # -----------------------------------------------
            # MATCHER RECORD
            # -----------------------------------------------

            prototype = {
                # Supabase catalogue UUID
                "id": str(jewellery["id"]),
                # Embedding
                "embedding": embedding,
                # Model
                "embedding_model": record.get("embedding_model"),
                # Dimension
                "embedding_dimension": record.get("embedding_dimension"),
                # Jewellery information
                "name": jewellery.get("name"),
                "design_code": jewellery.get("design_code"),
                "collection": jewellery.get("collection"),
                "type": jewellery.get("type"),
                "description": jewellery.get("description"),
                # Supabase image information
                "image_path": jewellery.get("image_path"),
                "image_url": jewellery.get("image_url"),
            }

            prototypes.append(prototype)

        print(
            "Valid Supabase prototypes:",
            len(prototypes),
        )

        return prototypes

    # ========================================================
    # MATCH IMAGE
    # ========================================================

    def match_image(
        self,
        image_path: str | Path,
        top_k: int = TOP_K,
    ):

        print()
        print("=" * 70)
        print("JEWELLERY IMAGE MATCHING")
        print("=" * 70)

        # ----------------------------------------------------
        # QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = self.generate_embedding(image_path)

        print(
            "Query embedding shape:",
            query_embedding.shape,
        )

        # ----------------------------------------------------
        # SUPABASE PROTOTYPE EMBEDDINGS
        # ----------------------------------------------------

        prototypes = self.get_prototype_embeddings()

        if not prototypes:

            print("No valid Supabase prototype embeddings found.")

            return []

        # ----------------------------------------------------
        # MATCH
        # ----------------------------------------------------

        results = self.matcher.match(
            query_embedding=query_embedding,
            prototypes=prototypes,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # MATCH INFORMATION
        # ----------------------------------------------------

        matching_results = []

        for result in results:

            matching_results.append(self.matcher.get_match_info(result))

        print()
        print("Matching completed.")
        print("Results:", len(matching_results))

        print("=" * 70)

        return matching_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing Jewellery Matching Service")

    service = JewelleryMatchingService()

    print()
    print("Matching service initialized successfully.")

    prototypes = service.get_prototype_embeddings()

    print()
    print("Supabase prototype count:", len(prototypes))

    if prototypes:

        print(
            "First prototype design code:",
            prototypes[0].get("design_code"),
        )

        print(
            "First prototype embedding dimension:",
            len(prototypes[0]["embedding"]),
        )
