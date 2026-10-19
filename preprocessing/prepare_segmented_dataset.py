# ============================================================
# PREPARE SEGMENTED JEWELLERY DATASET
# ============================================================
#
# Reads:
#
#   dataset/gold
#   dataset/prototypes
#
# Segments each jewellery image using the segmentation pipeline.
#
# Writes:
#
#   processed_segmented/gold
#   processed_segmented/prototypes
#
# Original images are NEVER modified.
#
# ============================================================

from pathlib import Path
import sys

from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_ROOT = PROJECT_ROOT / "dataset"

OUTPUT_ROOT = PROJECT_ROOT / "processed_segmented"

SOURCE_GOLD = SOURCE_ROOT / "gold"
SOURCE_PROTOTYPES = SOURCE_ROOT / "prototypes"

OUTPUT_GOLD = OUTPUT_ROOT / "gold"
OUTPUT_PROTOTYPES = OUTPUT_ROOT / "prototypes"


# ------------------------------------------------------------
# Supported image formats
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ------------------------------------------------------------
# Import segmentation
# ------------------------------------------------------------

BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from segmentation import prepare_ring_image

# ============================================================
# DIRECTORY SETUP
# ============================================================


def create_directories():

    OUTPUT_GOLD.mkdir(parents=True, exist_ok=True)

    OUTPUT_PROTOTYPES.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND IMAGES
# ============================================================


def get_images(directory: Path):

    if not directory.exists():

        print(f"\n[WARNING] Directory does not exist:\n" f"{directory}")

        return []

    images = [
        path
        for path in directory.rglob("*")
        if (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    ]

    return sorted(images, key=lambda path: str(path).lower())


# ============================================================
# PROCESS ONE IMAGE
# ============================================================


def process_image(input_path: Path, output_path: Path):

    try:

        image = Image.open(input_path).convert("RGB")

        result = prepare_ring_image(image)

        # ----------------------------------------------------
        # Save as JPEG.
        # ----------------------------------------------------

        result.save(output_path, format="JPEG", quality=95)

        return True

    except Exception as exc:

        print(f"\n[ERROR] Failed to process:\n" f"{input_path}\n" f"Reason: {exc}")

        return False


# ============================================================
# PROCESS DIRECTORY
# ============================================================


def process_directory(source_directory: Path, output_directory: Path, domain_name: str):

    images = get_images(source_directory)

    print("\n" + "=" * 70)
    print(f"PROCESSING {domain_name.upper()} DATA")
    print("=" * 70)

    print(f"Source: {source_directory}")

    print(f"Output: {output_directory}")

    print(f"Images found: {len(images)}")

    if not images:

        return {
            "total": 0,
            "success": 0,
            "failed": 0,
        }

    success_count = 0
    failed_count = 0

    for input_path in tqdm(images, desc=f"Segmenting {domain_name}"):

        # ----------------------------------------------------
        # Preserve filename.
        # ----------------------------------------------------

        output_filename = input_path.stem + ".jpg"

        output_path = output_directory / output_filename

        # ----------------------------------------------------
        # If duplicate filenames exist in different
        # subfolders, make the output name unique.
        # ----------------------------------------------------

        if output_path.exists():

            parent_name = input_path.parent.name

            output_filename = f"{parent_name}_{input_path.stem}.jpg"

            output_path = output_directory / output_filename

        success = process_image(input_path, output_path)

        if success:

            success_count += 1

        else:

            failed_count += 1

    return {
        "total": len(images),
        "success": success_count,
        "failed": failed_count,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("JEWELLERY SEGMENTED DATASET PREPARATION")
    print("=" * 70)

    print("\nOriginal images will NOT be modified.")

    print("\nSource:")

    print(f"  Gold:       {SOURCE_GOLD}")

    print(f"  Prototypes: {SOURCE_PROTOTYPES}")

    print("\nOutput:")

    print(f"  Gold:       {OUTPUT_GOLD}")

    print(f"  Prototypes: {OUTPUT_PROTOTYPES}")

    create_directories()

    # --------------------------------------------------------
    # Gold
    # --------------------------------------------------------

    gold_stats = process_directory(SOURCE_GOLD, OUTPUT_GOLD, "gold")

    # --------------------------------------------------------
    # Prototype
    # --------------------------------------------------------

    prototype_stats = process_directory(
        SOURCE_PROTOTYPES, OUTPUT_PROTOTYPES, "prototype"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total = gold_stats["total"] + prototype_stats["total"]

    success = gold_stats["success"] + prototype_stats["success"]

    failed = gold_stats["failed"] + prototype_stats["failed"]

    print("\n" + "=" * 70)
    print("SEGMENTED DATASET COMPLETE")
    print("=" * 70)

    print(
        f"\nGold:"
        f"\n  Total:   {gold_stats['total']}"
        f"\n  Success: {gold_stats['success']}"
        f"\n  Failed:  {gold_stats['failed']}"
    )

    print(
        f"\nPrototype:"
        f"\n  Total:   {prototype_stats['total']}"
        f"\n  Success: {prototype_stats['success']}"
        f"\n  Failed:  {prototype_stats['failed']}"
    )

    print(
        f"\nOverall:"
        f"\n  Total:   {total}"
        f"\n  Success: {success}"
        f"\n  Failed:  {failed}"
    )

    print(f"\nClean dataset saved at:" f"\n{OUTPUT_ROOT}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
