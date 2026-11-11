
import random

from PIL import Image

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from training.config import IMAGE_SIZE


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

# Standard ImageNet normalization.
# DINOv2 pretrained weights use ImageNet-style normalization.

NORMALIZE_MEAN = (
    0.485,
    0.456,
    0.406,
)

NORMALIZE_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def get_train_transform():

    return transforms.Compose([

        # Convert to RGB.
        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),

        # Resize while preserving jewellery proportions.
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),

        # Small rotation.
        transforms.RandomApply(
            [
                transforms.RandomRotation(
                    degrees=8,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=255,
                )
            ],
            p=0.5,
        ),

        # Small translation and scaling.
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=255,
                )
            ],
            p=0.4,
        ),

        # Brightness and contrast variation.
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                )
            ],
            p=0.5,
        ),

        # Occasional mild blur.
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0),
                )
            ],
            p=0.10,
        ),

        # Convert PIL image to tensor.
        transforms.ToTensor(),

        # Normalize for pretrained DINOv2.
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),

    ])


# ============================================================
# VALIDATION TRANSFORMATION
# ============================================================

def get_validation_transform():

    return transforms.Compose([

        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),

    ])


# ============================================================
# TWO AUGMENTED VIEWS
# ============================================================

class TwoViewTransform:

    def __init__(self):

        self.transform = get_train_transform()

    def __call__(self, image):

        # First augmented image.
        view1 = self.transform(
            image.copy()
        )

        # Second independently augmented image.
        view2 = self.transform(
            image.copy()
        )

        return view1, view2


# ============================================================
# TEST AUGMENTATION
# ============================================================

if __name__ == "__main__":

    from pathlib import Path

    from training.config import GOLD_DIR

    images = list(
        GOLD_DIR.glob("*")
    )

    images = [
        image
        for image in images
        if image.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        }
    ]

    if not images:

        print("No Gold images found.")

    else:

        image_path = images[0]

        print("Testing image:", image_path)

        image = Image.open(
            image_path
        ).convert("RGB")

        transform = TwoViewTransform()

        view1, view2 = transform(
            image
        )

        print()

        print("Original image size:", image.size)

        print("View 1 shape:", view1.shape)

        print("View 2 shape:", view2.shape)

        print("View 1 dtype:", view1.dtype)

        print("View 2 dtype:", view2.dtype)

        print()

        print("Augmentation test completed.")
