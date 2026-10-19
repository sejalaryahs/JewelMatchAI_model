import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIRS = {
    "gold": PROJECT_ROOT / "dataset" / "gold",
    "prototype": PROJECT_ROOT / "dataset" / "prototypes",
}

OUTPUT_DIR = PROJECT_ROOT / "processed"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# IMAGE UTILITIES
# ============================================================


def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def resize_keep_aspect(image, target_size=512):
    """
    Resize image while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise ValueError("Invalid image dimensions.")

    scale = target_size / max(height, width)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def pad_to_square(image, size=512):
    """
    Place image in a square canvas.
    """

    height, width = image.shape[:2]

    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255

    scale = min(
        size / width,
        size / height,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    x = (size - new_width) // 2
    y = (size - new_height) // 2

    canvas[y : y + new_height, x : x + new_width] = resized

    return canvas


# ============================================================
# SIMPLE FOREGROUND DETECTION
# ============================================================


def create_foreground_mask(image):
    """
    Attempts to identify the jewellery/foreground.

    This is intentionally conservative.

    The purpose is mainly to reduce large uniform backgrounds.
    It should not be considered a production-grade jewellery
    segmentation model.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Otsu threshold
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Try both polarities and choose the more useful mask
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size

    ratio = white_pixels / total_pixels

    if ratio > 0.70:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


def apply_mask(image, mask):
    """
    Applies foreground mask while keeping a clean white background.
    """

    background = np.ones_like(image) * 255

    foreground = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    inverse_mask = cv2.bitwise_not(mask)

    background_part = cv2.bitwise_and(
        background,
        background,
        mask=inverse_mask,
    )

    result = cv2.add(
        foreground,
        background_part,
    )

    return result


# ============================================================
# PROCESS ONE IMAGE
# ============================================================


def process_image(input_path, output_path):
    image = load_image(input_path)

    # Normalize large images
    image = resize_keep_aspect(
        image,
        target_size=512,
    )

    mask = create_foreground_mask(image)

    segmented = apply_mask(
        image,
        mask,
    )

    final_image = pad_to_square(
        segmented,
        size=512,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        final_image,
    )

    if not success:
        raise IOError(f"Could not save image: {output_path}")


# ============================================================
# PROCESS DATASET
# ============================================================


def process_directory(category):
    input_dir = INPUT_DIRS[category]
    output_dir = OUTPUT_DIR / category

    if not input_dir.exists():
        print(f"[WARNING] Directory does not exist: {input_dir}")
        return

    files = [
        file for file in input_dir.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print()
    print("=" * 70)
    print(f"PROCESSING: {category.upper()}")
    print("=" * 70)
    print(f"Images found: {len(files)}")

    for image_path in tqdm(
        files,
        desc=f"Processing {category}",
    ):
        relative_path = image_path.relative_to(input_dir)

        output_path = output_dir / relative_path

        try:
            process_image(
                image_path,
                output_path,
            )
        except Exception as exc:
            print(f"\n[ERROR] {image_path}: {exc}")


def main():
    print("=" * 70)
    print("JEWELLERY RING PREPROCESSING")
    print("=" * 70)

    for category in INPUT_DIRS:
        process_directory(category)

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
