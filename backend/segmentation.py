# ============================================================
# JEWELLERY IMAGE SEGMENTATION
# ============================================================
#
# Purpose:
#   Automatically isolate the jewellery from the background.
#
# Important:
#   - Does NOT depend on jewellery colour.
#   - Works for gold, green prototype, silver, etc.
#   - Uses GrabCut for foreground/background separation.
#   - Preserves internal holes/open areas in jewellery designs.
#   - Keeps the original image if segmentation confidence is low.
#
# ============================================================

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

MIN_IMAGE_SIZE = 80

# Percentage of image border used as definite background.
BORDER_RATIO = 0.03

# Minimum foreground area relative to complete image.
MIN_FOREGROUND_RATIO = 0.005

# Maximum foreground area relative to complete image.
MAX_FOREGROUND_RATIO = 0.90

# GrabCut iterations.
GRABCUT_ITERATIONS = 5


# ============================================================
# BASIC VALIDATION
# ============================================================


def _validate_image(image: Image.Image) -> Image.Image:
    """
    Make sure the input is a valid RGB PIL image.
    """

    if image is None:
        raise ValueError("Image is None.")

    image = image.convert("RGB")

    width, height = image.size

    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        raise ValueError(f"Image is too small for segmentation: {width}x{height}")

    return image


# ============================================================
# PIL -> OPENCV
# ============================================================


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB image to OpenCV BGR image.
    """

    rgb = np.array(image)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr


# ============================================================
# OPENCV -> PIL
# ============================================================


def _cv_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR image to PIL RGB image.
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb)


# ============================================================
# CREATE INITIAL GRABCUT MASK
# ============================================================


def _create_initial_mask(image: np.ndarray) -> np.ndarray:
    """
    Create an initial GrabCut mask.

    Assumption:
        Jewellery is generally somewhere inside the image.

    Border:
        Definite background.

    Centre:
        Probable foreground.

    This is deliberately colour-independent.
    """

    height, width = image.shape[:2]

    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

    # --------------------------------------------------------
    # Definite background around image border
    # --------------------------------------------------------

    border_x = max(1, int(width * BORDER_RATIO))
    border_y = max(1, int(height * BORDER_RATIO))

    mask[:border_y, :] = cv2.GC_BGD
    mask[-border_y:, :] = cv2.GC_BGD
    mask[:, :border_x] = cv2.GC_BGD
    mask[:, -border_x:] = cv2.GC_BGD

    # --------------------------------------------------------
    # Centre region = probable foreground
    #
    # We intentionally use a generous region because rings
    # can be tilted or occupy different parts of the image.
    # --------------------------------------------------------

    x1 = int(width * 0.12)
    y1 = int(height * 0.12)

    x2 = int(width * 0.88)
    y2 = int(height * 0.88)

    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # Centre-most area gets stronger foreground probability.
    # --------------------------------------------------------

    cx1 = int(width * 0.25)
    cy1 = int(height * 0.20)

    cx2 = int(width * 0.75)
    cy2 = int(height * 0.80)

    mask[cy1:cy2, cx1:cx2] = cv2.GC_PR_FGD

    return mask


# ============================================================
# GRABCUT
# ============================================================


def _run_grabcut(image: np.ndarray) -> np.ndarray:
    """
    Run GrabCut using a mask initialized from image geometry.
    """

    mask = _create_initial_mask(image)

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        None,
        background_model,
        foreground_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK,
    )

    # --------------------------------------------------------
    # GrabCut output:
    #
    # GC_FGD     = definite foreground
    # GC_PR_FGD  = probable foreground
    #
    # Both are treated as jewellery.
    # --------------------------------------------------------

    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return foreground_mask


# ============================================================
# MORPHOLOGICAL CLEANUP
# ============================================================


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove tiny noise while keeping the jewellery structure.
    """

    height, width = mask.shape[:2]

    kernel_size = max(3, int(min(height, width) * 0.01))

    # Make kernel odd.
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close very small breaks.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove tiny isolated noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


# ============================================================
# KEEP RELEVANT FOREGROUND
# ============================================================


def _keep_relevant_components(mask: np.ndarray) -> np.ndarray:
    """
    Keep the main jewellery component WITHOUT filling its
    internal holes.

    This is important for jewellery because:
        - decorative cut-outs are part of the design
        - ring openings must remain background
        - engraved/open patterns must be preserved

    We therefore work directly with the existing binary mask
    instead of drawing a filled contour.
    """

    height, width = mask.shape[:2]

    # --------------------------------------------------------
    # Find connected components directly from the mask.
    #
    # Unlike cv2.drawContours(..., FILLED), this preserves
    # holes that already exist inside the foreground.
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    image_area = float(height * width)

    components = []

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        ratio = area / image_area

        # Ignore extremely tiny noise.
        if ratio < 0.001:
            continue

        components.append((label, area))

    if not components:
        return mask

    # --------------------------------------------------------
    # Largest connected jewellery component.
    # --------------------------------------------------------

    components.sort(key=lambda item: item[1], reverse=True)

    largest_label, largest_area = components[0]

    # --------------------------------------------------------
    # Create output mask using the ORIGINAL pixels.
    #
    # This is the important part:
    #
    # We copy the original mask pixels belonging to the
    # component instead of filling its outer contour.
    #
    # Therefore internal holes remain holes.
    # --------------------------------------------------------

    result = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # --------------------------------------------------------
    # Occasionally a jewellery item can consist of multiple
    # touching/near-touching components.
    #
    # Keep reasonably large components close to the main one.
    # --------------------------------------------------------

    for label, area in components[1:]:

        if area < largest_area * 0.08:
            continue

        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)

        # Bounding box of this component.
        coords = cv2.findNonZero(component_mask)

        if coords is None:
            continue

        x, y, w, h = cv2.boundingRect(coords)

        # Centre of component.
        component_cx = x + (w / 2)
        component_cy = y + (h / 2)

        # Bounding box of main component.
        main_x = stats[largest_label, cv2.CC_STAT_LEFT]
        main_y = stats[largest_label, cv2.CC_STAT_TOP]
        main_w = stats[largest_label, cv2.CC_STAT_WIDTH]
        main_h = stats[largest_label, cv2.CC_STAT_HEIGHT]

        main_cx = main_x + (main_w / 2)
        main_cy = main_y + (main_h / 2)

        # Distance between component centres.
        distance = np.sqrt(
            (component_cx - main_cx) ** 2 + (component_cy - main_cy) ** 2
        )

        max_distance = max(width, height) * 0.35

        if distance <= max_distance:

            result[component_mask > 0] = 255

    return result


# ============================================================
# VALIDATE SEGMENTATION
# ============================================================


def _segmentation_is_reasonable(mask: np.ndarray) -> bool:
    """
    Check whether segmentation produced a plausible amount
    of foreground.
    """

    foreground_pixels = np.count_nonzero(mask)

    total_pixels = mask.shape[0] * mask.shape[1]

    ratio = foreground_pixels / float(total_pixels)

    if ratio < MIN_FOREGROUND_RATIO:
        return False

    if ratio > MAX_FOREGROUND_RATIO:
        return False

    return True


# ============================================================
# CROP USING MASK
# ============================================================


def _crop_to_foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop the image around the detected jewellery.

    A small amount of padding is retained so that the jewellery
    does not touch the image boundary.
    """

    coordinates = cv2.findNonZero(mask)

    if coordinates is None:
        return image

    x, y, width, height = cv2.boundingRect(coordinates)

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Add padding.
    # --------------------------------------------------------

    padding_x = max(5, int(width * 0.08))
    padding_y = max(5, int(height * 0.08))

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(image_width, x + width + padding_x)

    y2 = min(image_height, y + height + padding_y)

    return image[y1:y2, x1:x2]


# ============================================================
# APPLY MASK TO IMAGE
# ============================================================


def _apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply segmentation mask.

    Background becomes a neutral light background.

    IMPORTANT:
        We do not fill internal holes in the jewellery.
        Those holes are part of the design.
    """

    foreground = cv2.bitwise_and(image, image, mask=mask)

    # --------------------------------------------------------
    # Neutral background.
    # --------------------------------------------------------

    background = np.full_like(image, 245, dtype=np.uint8)

    inverse_mask = cv2.bitwise_not(mask)

    background_part = cv2.bitwise_and(background, background, mask=inverse_mask)

    result = cv2.add(foreground, background_part)

    return result


# ============================================================
# MAIN SEGMENTATION FUNCTION
# ============================================================


def prepare_ring_image(image: Image.Image, return_debug: bool = False):
    """
    Main jewellery segmentation function.

    Parameters
    ----------
    image:
        PIL RGB image.

    return_debug:
        If True, returns additional segmentation information.

    Returns
    -------
    PIL.Image

        Segmented/cropped jewellery image.

    If return_debug=True:
        {
            "original": PIL.Image,
            "mask": PIL.Image,
            "segmented": PIL.Image,
            "success": bool
        }
    """

    original = _validate_image(image)

    cv_image = _pil_to_cv(original)

    # --------------------------------------------------------
    # Run GrabCut.
    # --------------------------------------------------------

    mask = _run_grabcut(cv_image)

    # --------------------------------------------------------
    # Clean mask.
    # --------------------------------------------------------

    mask = _clean_mask(mask)

    # --------------------------------------------------------
    # Keep relevant jewellery components.
    # --------------------------------------------------------

    mask = _keep_relevant_components(mask)

    # --------------------------------------------------------
    # Validate result.
    # --------------------------------------------------------

    success = _segmentation_is_reasonable(mask)

    if not success:

        # ----------------------------------------------------
        # Safety fallback:
        #
        # If automatic segmentation looks unreliable, return
        # the original image rather than destroying the ring.
        # ----------------------------------------------------

        segmented = cv_image.copy()

    else:

        # ----------------------------------------------------
        # Apply mask.
        # ----------------------------------------------------

        segmented = _apply_mask(cv_image, mask)

        # ----------------------------------------------------
        # Crop around jewellery.
        # ----------------------------------------------------

        segmented = _crop_to_foreground(segmented, mask)

    segmented_pil = _cv_to_pil(segmented)

    if not return_debug:
        return segmented_pil

    mask_pil = Image.fromarray(mask)

    return {
        "original": original,
        "mask": mask_pil,
        "segmented": segmented_pil,
        "success": success,
    }


# ============================================================
# TESTING FUNCTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("JEWELLERY SEGMENTATION TEST")
    print("=" * 70)

    test_path = (
        input("\nEnter the full path of a jewellery image:\n> ").strip().strip('"')
    )

    if not test_path:

        print("\nNo image path supplied.")
        raise SystemExit(1)

    try:

        image = Image.open(test_path).convert("RGB")

        result = prepare_ring_image(image, return_debug=True)

        print(f"\nSegmentation successful: " f"{result['success']}")

        original_path = "segmentation_original.jpg"
        mask_path = "segmentation_mask.png"
        result_path = "segmentation_result.jpg"

        result["original"].save(original_path)
        result["mask"].save(mask_path)
        result["segmented"].save(result_path)

        print("\nFiles created:")
        print(f"Original:    {original_path}")
        print(f"Mask:        {mask_path}")
        print(f"Segmented:   {result_path}")

        print("\nSegmentation test completed.")

    except Exception as exc:

        print("\nERROR:")
        print(exc)
        raise
