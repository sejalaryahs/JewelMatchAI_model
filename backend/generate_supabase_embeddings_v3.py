import gc
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from backend.config import IMAGE_SIZE
from backend.database.supabase import (
    supabase,
    SUPABASE_STORAGE_BUCKET,
)
from backend.model_loader import ModelLoader
from backend.segmentation import segment_jewellery
from training.augmentations import get_validation_transform

# ============================================================
# CONFIGURATION
# ============================================================

JEWELLERY_TABLE = "jewellery"

EMBEDDINGS_TABLE = "jewellery_embeddings_v3"

MODEL_VERSION = "jewellery_dinov2_v3"

EMBEDDING_DIMENSION = 512

COLLECTIONS_TO_PROCESS = [
    "Gold",
    "Prototype",
]

# Supabase catalogue pagination.
PAGE_SIZE = 100

# Process one image at a time.
# This is suitable for your current CPU environment.
BATCH_SIZE = 1


# ============================================================
# DEVICE
# ============================================================


def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print("Using device:", device)

    return device


# ============================================================
# LOAD MODEL
# ============================================================


def load_model():

    print()
    print("=" * 60)
    print("Loading Model 3")
    print("=" * 60)

    loader = ModelLoader()

    model = loader.load_model()

    device = get_device()

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    print("Embedding dimension:", EMBEDDING_DIMENSION)

    return model, device


# ============================================================
# LOAD VALIDATION TRANSFORM
# ============================================================


def load_transform():

    print()
    print("Loading validation transformation...")

    transform = get_validation_transform()

    print("Validation transformation loaded.")

    return transform


# ============================================================
# FETCH JEWELLERY RECORDS
# ============================================================


def fetch_jewellery_records():

    print()
    print("=" * 60)
    print("Fetching jewellery records from Supabase")
    print("=" * 60)

    records = []

    start = 0

    while True:

        end = start + PAGE_SIZE - 1

        response = (
            supabase.table(JEWELLERY_TABLE)
            .select(
                "id,"
                "name,"
                "collection,"
                "type,"
                "image_path,"
                "image_url,"
                "design_code"
            )
            .in_(
                "collection",
                COLLECTIONS_TO_PROCESS,
            )
            .order("id")
            .range(start, end)
            .execute()
        )

        page_records = response.data or []

        if not page_records:

            break

        records.extend(page_records)

        print(f"Fetched {len(page_records)} records " f"(total: {len(records)})")

        if len(page_records) < PAGE_SIZE:

            break

        start += PAGE_SIZE

    print()

    print("Total jewellery records:", len(records))

    gold_count = sum(1 for record in records if record.get("collection") == "Gold")

    prototype_count = sum(
        1 for record in records if record.get("collection") == "Prototype"
    )

    print("Gold records:", gold_count)

    print("Prototype records:", prototype_count)

    return records


# ============================================================
# DOWNLOAD IMAGE FROM SUPABASE STORAGE
# ============================================================


def download_image(image_path):

    if not image_path:

        raise ValueError("Jewellery record does not contain image_path.")

    print("Downloading:", image_path)

    response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(image_path)

    if not response:

        raise ValueError(f"Empty image response: {image_path}")

    return response


# ============================================================
# PREPARE IMAGE
# ============================================================


def prepare_image(image_bytes):

    with tempfile.TemporaryDirectory(
        prefix="jewellery_embedding_v3_"
    ) as temp_directory:

        temp_directory = Path(temp_directory)

        original_path = temp_directory / "original_image.jpg"

        segmented_path = temp_directory / "segmented_image.jpg"

        # ----------------------------------------------------
        # Save downloaded image.
        # ----------------------------------------------------

        with open(original_path, "wb") as file:

            file.write(image_bytes)

        # ----------------------------------------------------
        # Read image and apply EXIF orientation.
        # ----------------------------------------------------

        with Image.open(original_path) as image:

            image = ImageOps.exif_transpose(image)

            image = image.convert("RGB")

            image.save(
                original_path,
                format="JPEG",
                quality=95,
            )

        # ----------------------------------------------------
        # Existing jewellery segmentation.
        # ----------------------------------------------------

        segment_jewellery(
            input_path=original_path,
            output_path=segmented_path,
        )

        # ----------------------------------------------------
        # Read segmented image.
        # ----------------------------------------------------

        with Image.open(segmented_path) as image:

            image = ImageOps.exif_transpose(image)

            image = image.convert("RGB")

            image = image.copy()

        # ----------------------------------------------------
        # Ensure expected image size.
        # ----------------------------------------------------

        if image.size != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            raise ValueError(
                "Segmented image has incorrect size: "
                f"{image.size}. "
                f"Expected {(IMAGE_SIZE, IMAGE_SIZE)}."
            )

        return image


# ============================================================
# GENERATE EMBEDDING
# ============================================================


def generate_embedding(
    model,
    device,
    image,
    transform,
):

    # --------------------------------------------------------
    # Apply the exact validation transform.
    # --------------------------------------------------------

    tensor = transform(image)

    # Expected:
    # [3, 224, 224]

    if tensor.shape != (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):

        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    # --------------------------------------------------------
    # Add batch dimension.
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # --------------------------------------------------------
    # Model inference.
    # --------------------------------------------------------

    with torch.inference_mode():

        output = model(tensor)

    # --------------------------------------------------------
    # Convert model output to a 1D vector.
    # --------------------------------------------------------

    if isinstance(output, dict):

        if "embedding" in output:

            output = output["embedding"]

        else:

            raise ValueError(
                "Model returned a dictionary " "without an 'embedding' key."
            )

    if isinstance(output, tuple):

        output = output[0]

    if not isinstance(output, torch.Tensor):

        raise TypeError("Model output is not a torch.Tensor.")

    embedding = output.detach().float().cpu().numpy().reshape(-1)

    # --------------------------------------------------------
    # Validate embedding dimension.
    # --------------------------------------------------------

    if embedding.shape != (EMBEDDING_DIMENSION,):

        raise ValueError(
            "Incorrect embedding dimension: "
            f"{embedding.shape}. "
            f"Expected ({EMBEDDING_DIMENSION},)."
        )

    # --------------------------------------------------------
    # Validate finite values.
    # --------------------------------------------------------

    if not np.isfinite(embedding).all():

        raise ValueError("Embedding contains NaN or infinite values.")

    # --------------------------------------------------------
    # L2 normalize.
    # --------------------------------------------------------

    norm = np.linalg.norm(embedding)

    if norm <= 0:

        raise ValueError("Embedding norm is zero.")

    embedding = embedding / norm

    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    final_norm = np.linalg.norm(embedding)

    if not np.isclose(
        final_norm,
        1.0,
        atol=1e-5,
    ):

        raise ValueError(f"Embedding normalization failed: {final_norm}")

    return embedding.astype(np.float32)


# ============================================================
# SAVE EMBEDDING TO SUPABASE
# ============================================================


def save_embedding(
    jewellery_id,
    embedding,
):

    if embedding.shape != (EMBEDDING_DIMENSION,):

        raise ValueError("Cannot save embedding with incorrect dimension.")

    if not np.isfinite(embedding).all():

        raise ValueError("Cannot save embedding containing invalid values.")

    embedding_list = embedding.tolist()

    updated_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "jewellery_id": str(jewellery_id),
        "embedding_model": MODEL_VERSION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding": embedding_list,
        "updated_at": updated_at,
    }

    response = (
        supabase.table(EMBEDDINGS_TABLE)
        .upsert(
            payload,
            on_conflict="jewellery_id",
        )
        .execute()
    )

    return response


# ============================================================
# PROCESS SINGLE JEWELLERY RECORD
# ============================================================


def process_record(
    record,
    model,
    device,
    transform,
):

    jewellery_id = record.get("id")

    name = record.get("name", "Unnamed")

    collection = record.get("collection", "Unknown")

    image_path = record.get("image_path")

    print()
    print("-" * 60)

    print("Jewellery ID:", jewellery_id)

    print("Name:", name)

    print("Collection:", collection)

    print("Image path:", image_path)

    if not jewellery_id:

        raise ValueError("Jewellery record is missing its ID.")

    if not image_path:

        raise ValueError("Jewellery record is missing image_path.")

    # --------------------------------------------------------
    # Download image.
    # --------------------------------------------------------

    image_bytes = download_image(image_path)

    # --------------------------------------------------------
    # Segment image.
    # --------------------------------------------------------

    image = prepare_image(image_bytes)

    # --------------------------------------------------------
    # Generate embedding.
    # --------------------------------------------------------

    embedding = generate_embedding(
        model=model,
        device=device,
        image=image,
        transform=transform,
    )

    # --------------------------------------------------------
    # Save embedding.
    # --------------------------------------------------------

    save_embedding(
        jewellery_id=jewellery_id,
        embedding=embedding,
    )

    print("Embedding saved successfully.")

    print("Dimension:", embedding.shape)

    print(
        "Embedding norm:",
        float(np.linalg.norm(embedding)),
    )

    print("Collection:", collection)

    return True


# ============================================================
# VERIFY SAVED EMBEDDING
# ============================================================


def verify_embedding(jewellery_id):

    print()
    print("Verifying saved embedding...")

    response = (
        supabase.table(EMBEDDINGS_TABLE)
        .select("jewellery_id," "embedding_model," "embedding_dimension," "embedding")
        .eq(
            "jewellery_id",
            str(jewellery_id),
        )
        .limit(1)
        .execute()
    )

    records = response.data or []

    if not records:

        raise ValueError("Embedding was not found in Supabase.")

    record = records[0]

    embedding = np.asarray(
        record["embedding"],
        dtype=np.float32,
    )

    print("Saved jewellery ID:", record["jewellery_id"])

    print("Saved model:", record["embedding_model"])

    print("Saved dimension:", record["embedding_dimension"])

    print("Retrieved shape:", embedding.shape)

    print(
        "Retrieved norm:",
        float(np.linalg.norm(embedding)),
    )

    if embedding.shape != (EMBEDDING_DIMENSION,):

        raise ValueError("Saved embedding has incorrect dimension.")

    if not np.isfinite(embedding).all():

        raise ValueError("Saved embedding contains invalid values.")

    if not np.isclose(
        np.linalg.norm(embedding),
        1.0,
        atol=1e-5,
    ):

        raise ValueError("Saved embedding is not normalized.")

    print("Verification successful.")

    return True


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)

    print("MODEL 3 - SUPABASE EMBEDDING GENERATOR")

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    model, device = load_model()

    # --------------------------------------------------------
    # Load validation transform.
    # --------------------------------------------------------

    transform = load_transform()

    # --------------------------------------------------------
    # Fetch catalogue records.
    # --------------------------------------------------------

    records = fetch_jewellery_records()

    if not records:

        print()

        print("No Gold or Prototype records found.")

        return

    # --------------------------------------------------------
    # Process records.
    # --------------------------------------------------------

    successful = 0

    failed = 0

    total = len(records)

    first_successful_id = None

    print()

    print("=" * 70)

    print("STARTING EMBEDDING GENERATION")

    print("=" * 70)

    for index, record in enumerate(
        records,
        start=1,
    ):

        print()

        print(f"Processing {index}/{total}")

        try:

            process_record(
                record=record,
                model=model,
                device=device,
                transform=transform,
            )

            successful += 1

            if first_successful_id is None:

                first_successful_id = record["id"]

        except Exception as error:

            failed += 1

            print()

            print(
                "ERROR processing record:",
                record.get("id"),
            )

            print("Error:", error)

        finally:

            gc.collect()

            if device.type == "cuda":

                torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("EMBEDDING GENERATION SUMMARY")

    print("=" * 70)

    print("Total records:", total)

    print("Successful:", successful)

    print("Failed:", failed)

    print("Model:", MODEL_VERSION)

    print("Embedding dimension:", EMBEDDING_DIMENSION)

    # --------------------------------------------------------
    # Verify one embedding.
    # --------------------------------------------------------

    if first_successful_id is not None:

        try:

            verify_embedding(first_successful_id)

        except Exception as error:

            print()

            print(
                "Verification error:",
                error,
            )

    print()

    print("=" * 70)

    print("PROCESS COMPLETED")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
