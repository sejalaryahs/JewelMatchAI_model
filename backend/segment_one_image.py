import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# SEGMENT ONE JEWELLERY IMAGE ONLY
# --------------------------------------------------

IMAGE_PATH = Path(
    r"E:/sejal/Jewellery_Image_Training/dataset/prototypes/WhatsApp Image 2026-09-11 at 17.04.48 (1).jpeg"
)
OUTPUT_PATH = IMAGE_PATH.with_name("ring_segmented.png")

BRUSH_SIZE = 9

# GrabCut labels
BACKGROUND = cv2.GC_BGD
FOREGROUND = cv2.GC_FGD
PROB_BACKGROUND = cv2.GC_PR_BGD
PROB_FOREGROUND = cv2.GC_PR_FGD

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Update IMAGE_PATH with the actual image location."
    )

original = image.copy()
height, width = image.shape[:2]

# Initially, pixels are probable background.
mask = np.full((height, width), PROB_BACKGROUND, dtype=np.uint8)

# Keep track of manually marked foreground/background pixels.
foreground_marks = np.zeros((height, width), dtype=np.uint8)
background_marks = np.zeros((height, width), dtype=np.uint8)

drawing = False
current_label = FOREGROUND


def draw_marks(event, x, y, flags, param):
    global drawing, current_label

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_label = FOREGROUND

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        current_label = BACKGROUND

    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        drawing = False
        paint(x, y, current_label)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        paint(x, y, current_label)


def paint(x, y, label):
    radius = BRUSH_SIZE

    if label == FOREGROUND:
        cv2.circle(foreground_marks, (x, y), radius, 255, -1)
        cv2.circle(background_marks, (x, y), radius, 0, -1)
    else:
        cv2.circle(background_marks, (x, y), radius, 255, -1)
        cv2.circle(foreground_marks, (x, y), radius, 0, -1)

    refresh_display()


def refresh_display():
    display = original.copy()

    # Green strokes indicate jewellery.
    display[foreground_marks > 0] = (0, 255, 0)

    # Red strokes indicate unwanted material/background.
    display[background_marks > 0] = (0, 0, 255)

    cv2.imshow("Mark Jewellery", display)


print("\nINSTRUCTIONS")
print("1. LEFT-CLICK and drag over solid green jewellery.")
print("2. RIGHT-CLICK and drag over the outside background.")
print("3. Also mark brown material in the decorative openings")
print("   and the brown pieces at the top/bottom as background.")
print("4. Mark a few separate areas of the green jewellery.")
print("5. Press G to run segmentation.")
print("6. Press Q to quit without saving.\n")

cv2.namedWindow("Mark Jewellery", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Mark Jewellery", draw_marks)
refresh_display()

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()
        print("Cancelled. No output saved.")
        raise SystemExit

    if key == ord("g"):
        break

cv2.destroyAllWindows()

# Convert user markings into GrabCut labels.
mask[foreground_marks > 0] = FOREGROUND
mask[background_marks > 0] = BACKGROUND

if np.count_nonzero(foreground_marks) == 0:
    raise ValueError("Mark some green jewellery with the LEFT mouse button.")

if np.count_nonzero(background_marks) == 0:
    raise ValueError("Mark some background with the RIGHT mouse button.")

# Refine the manually marked mask.
background_model = np.zeros((1, 65), dtype=np.float64)
foreground_model = np.zeros((1, 65), dtype=np.float64)

cv2.grabCut(
    original, mask, None, background_model, foreground_model, 8, cv2.GC_INIT_WITH_MASK
)

# Extract foreground pixels.
result_mask = np.where((mask == FOREGROUND) | (mask == PROB_FOREGROUND), 255, 0).astype(
    np.uint8
)

# Restore all pixels manually marked as jewellery.
result_mask[foreground_marks > 0] = 255

# Ensure explicitly marked background stays transparent.
result_mask[background_marks > 0] = 0

# Create transparent PNG.
rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = result_mask

if not cv2.imwrite(str(OUTPUT_PATH), rgba):
    raise RuntimeError(f"Could not save output: {OUTPUT_PATH}")

# Save a white-background preview for easier inspection.
white_background = np.full_like(original, 255)
alpha = result_mask.astype(np.float32) / 255.0

preview = (
    original.astype(np.float32) * alpha[:, :, None]
    + white_background.astype(np.float32) * (1 - alpha[:, :, None])
).astype(np.uint8)

preview_path = IMAGE_PATH.with_name("ring_segmented_preview.jpg")
cv2.imwrite(str(preview_path), preview)

cv2.imshow("Segmented Jewellery Preview", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\nTransparent image saved: {OUTPUT_PATH}")
print(f"White-background preview saved: {preview_path}")
print("Original image was not modified.")
