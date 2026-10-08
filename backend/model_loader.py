from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2.onnx"


# ============================================================
# IMAGE PREPROCESSING
#
# Must match the preprocessing used during training:
#
# Resize 256
# Center Crop 224
# RGB
# ToTensor
# ImageNet normalization
# ============================================================

IMAGE_SIZE = 224
RESIZE_SIZE = 256

IMAGE_MEAN = np.array(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=np.float32,
)

IMAGE_STD = np.array(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=np.float32,
)


# ============================================================
# ONNX MODEL LOADER
# ============================================================


class ModelLoader:

    def __init__(self):

        print()
        print("=" * 70)
        print("LOADING JEWELLERY DINOv2 ONNX")
        print("=" * 70)

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            raise FileNotFoundError(f"ONNX model not found:\n{MODEL_PATH}")

        print(f"Model: {MODEL_PATH}")

        # ----------------------------------------------------
        # ONNX Runtime
        # ----------------------------------------------------

        print()
        print("Loading ONNX Runtime...")

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        # ----------------------------------------------------
        # Input / output names
        # ----------------------------------------------------

        self.input_name = self.session.get_inputs()[0].name

        self.output_name = self.session.get_outputs()[0].name

        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_name}")

        print()
        print("✓ ONNX model loaded successfully")

        print()
        print("=" * 70)
        print("DINOv2 ONNX READY")
        print("=" * 70)

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    def preprocess(self, image):

        # ----------------------------------------------------
        # Convert to PIL Image
        # ----------------------------------------------------

        if not isinstance(image, Image.Image):

            image = Image.fromarray(image)

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Resize
        #
        # torchvision.transforms.Resize(256) with an integer
        # keeps the aspect ratio and makes the shorter side
        # equal to 256.
        # ----------------------------------------------------

        width, height = image.size

        if width < height:

            new_width = RESIZE_SIZE

            new_height = int(round(height * RESIZE_SIZE / width))

        else:

            new_height = RESIZE_SIZE

            new_width = int(round(width * RESIZE_SIZE / height))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        # ----------------------------------------------------
        # Center Crop 224 x 224
        # ----------------------------------------------------

        width, height = image.size

        left = (width - IMAGE_SIZE) // 2
        top = (height - IMAGE_SIZE) // 2

        right = left + IMAGE_SIZE
        bottom = top + IMAGE_SIZE

        image = image.crop((left, top, right, bottom))

        # ----------------------------------------------------
        # Convert to NumPy
        #
        # PIL:
        #   H x W x C
        #
        # ONNX:
        #   C x H x W
        # ----------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # ToTensor equivalent
        #
        # torchvision ToTensor() converts:
        #
        # uint8 0-255
        #
        # into:
        #
        # float32 0-1
        # ----------------------------------------------------

        image_array = image_array / 255.0

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        image_array = (image_array - IMAGE_MEAN) / IMAGE_STD

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image_array = np.transpose(
            image_array,
            (2, 0, 1),
        )

        # ----------------------------------------------------
        # Add batch dimension
        #
        # C x H x W
        #       ↓
        # 1 x C x H x W
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return image_array.astype(np.float32)

    # ========================================================
    # GET EMBEDDING
    # ========================================================

    def get_embedding(self, image):

        pixel_values = self.preprocess(image)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: pixel_values},
        )

        embedding = outputs[0][0]

        # ----------------------------------------------------
        # Safety normalization
        # ----------------------------------------------------

        norm = np.linalg.norm(embedding)

        if norm > 0:

            embedding = embedding / norm

        return embedding.astype(np.float32)
