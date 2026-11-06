from pathlib import Path
from functools import lru_cache

from PIL import Image, ImageOps

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

GOLD_DIR = DATASET_DIR / "gold"

PROTOTYPE_DIR = DATASET_DIR / "prototypes"

SEGMENTED_DIR = DATASET_DIR / "segmented"

GOLD_OUTPUT_DIR = SEGMENTED_DIR / "gold"

PROTOTYPE_OUTPUT_DIR = SEGMENTED_DIR / "prototypes"


# ============================================================
# SEGMENTATION CONFIGURATION
# ============================================================

MODEL_NAME = "u2netp"

IMAGE_SIZE = 224

BACKGROUND_COLOR = (255, 255, 255)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

GOLD_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOTYPE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMAGE UTILITIES
# ============================================================


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory: Path) -> list[Path]:

    if not directory.exists():

        raise FileNotFoundError(f"Directory does not exist: {directory}")

    images = []

    for path in directory.rglob("*"):

        if is_image_file(path):

            images.append(path)

    return sorted(images)


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================


@lru_cache(maxsize=1)
def get_segmentation_session():

    from rembg import new_session

    print()
    print("Loading segmentation model...")
    print("Model:", MODEL_NAME)

    session = new_session(MODEL_NAME)

    print("Segmentation model loaded successfully.")

    return session


# ============================================================
# IMAGE RESIZING
# ============================================================


def resize_and_pad(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    background_color=BACKGROUND_COLOR,
) -> Image.Image:

    image = image.convert("RGB")

    image.thumbnail(
        (image_size, image_size),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        background_color,
    )

    x = (image_size - image.width) // 2

    y = (image_size - image.height) // 2

    canvas.paste(
        image,
        (x, y),
    )

    return canvas


# ============================================================
# PREPARE IMAGE
# ============================================================


def prepare_image(input_path: str | Path) -> Image.Image:

    input_path = Path(input_path)

    if not input_path.exists():

        raise FileNotFoundError(f"Input image not found: {input_path}")

    with Image.open(input_path) as image:

        # Correct image orientation using EXIF metadata.
        image = ImageOps.exif_transpose(image)

        # Convert image to RGB.
        image = image.convert("RGB")

        return image.copy()


# ============================================================
# SEGMENT JEWELLERY
# ============================================================


def segment_jewellery(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:

    input_path = Path(input_path)

    output_path = Path(output_path)

    if not input_path.exists():

        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load original image.
    original = prepare_image(input_path)

    # Reuse the segmentation model.
    session = get_segmentation_session()

    # Import rembg only when required.
    from rembg import remove

    # Remove background.
    result = remove(
        original,
        session=session,
    )

    # Convert output to RGBA.
    result = result.convert("RGBA")

    # Create white background.
    background = Image.new(
        "RGBA",
        result.size,
        (
            BACKGROUND_COLOR[0],
            BACKGROUND_COLOR[1],
            BACKGROUND_COLOR[2],
            255,
        ),
    )

    # Composite jewellery onto white background.
    background.alpha_composite(result)

    # Convert to RGB.
    final_image = background.convert("RGB")

    # Resize and pad.
    final_image = resize_and_pad(
        final_image,
        IMAGE_SIZE,
    )

    # Save processed image.
    final_image.save(
        output_path,
        format="JPEG",
        quality=95,
    )

    return output_path


# ============================================================
# PROCESS DATASET DIRECTORY
# ============================================================


def process_directory(
    input_directory: Path,
    output_directory: Path,
):

    input_directory = Path(input_directory)

    output_directory = Path(output_directory)

    images = list_images(input_directory)

    print()
    print("=" * 60)

    print("JEWELLERY SEGMENTATION")

    print("=" * 60)

    print("Input:", input_directory)

    print("Output:", output_directory)

    print("Images found:", len(images))

    print()

    success_count = 0

    failed_count = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        try:

            # Preserve relative folder structure.
            relative_path = image_path.relative_to(input_directory)

            # Save segmented images as JPEG.
            output_path = output_directory / relative_path.with_suffix(".jpg")

            # Segment image.
            segment_jewellery(
                image_path,
                output_path,
            )

            success_count += 1

            print(f"[{index}/{len(images)}] " f"SUCCESS: {image_path.name}")

        except Exception as error:

            failed_count += 1

            print(f"[{index}/{len(images)}] " f"FAILED: {image_path.name}")

            print("Reason:", error)

    print()

    print("SEGMENTATION COMPLETED")

    print("Successful:", success_count)

    print("Failed:", failed_count)

    print("=" * 60)

    return success_count, failed_count


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Jewellery Segmentation Module")

    print("Project:", PROJECT_ROOT)

    print()
    print("Gold dataset:", GOLD_DIR)

    print("Prototype dataset:", PROTOTYPE_DIR)

    print()
    print("Segmented gold:", GOLD_OUTPUT_DIR)

    print("Segmented prototypes:", PROTOTYPE_OUTPUT_DIR)

    print()
    print("This module provides segmentation functions.")
    print("Use the dataset segmentation script to process images.")
