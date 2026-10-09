from pathlib import Path
import gc

import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2.onnx"

IMAGE_SIZE = 224
RESIZE_SIZE = 256

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelLoader:
    def __init__(self):
        print("=" * 70)
        print("LOADING JEWELMATCH DINOv2 ONNX")
        print("=" * 70)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

        # Restrict CPU threading and use conservative memory settings.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model: {MODEL_PATH}")
        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_name}")
        print("ONNX model loaded successfully.")

    def preprocess(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        width, height = image.size

        if width < height:
            new_width = RESIZE_SIZE
            new_height = int(round(height * RESIZE_SIZE / width))
        else:
            new_height = RESIZE_SIZE
            new_width = int(round(width * RESIZE_SIZE / height))

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        left = (new_width - IMAGE_SIZE) // 2
        top = (new_height - IMAGE_SIZE) // 2

        cropped = resized.crop(
            (
                left,
                top,
                left + IMAGE_SIZE,
                top + IMAGE_SIZE,
            )
        )

        # Create a single normalized float32 input tensor.
        image_array = np.asarray(cropped, dtype=np.float32)
        image_array *= 1.0 / 255.0
        image_array -= IMAGE_MEAN
        image_array /= IMAGE_STD
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.ascontiguousarray(image_array, dtype=np.float32)

        # Drop temporary PIL references.
        del image, resized, cropped

        return image_array

    def get_embedding(self, image):
        pixel_values = None
        outputs = None
        embedding = None

        try:
            pixel_values = self.preprocess(image)

            outputs = self.session.run(
                [self.output_name],
                {self.input_name: pixel_values},
            )

            embedding = np.array(
                outputs[0][0],
                dtype=np.float32,
                copy=True,
            )

            norm = float(np.linalg.norm(embedding))

            if norm > 0:
                embedding /= norm

            return embedding

        finally:
            # Release Python references to temporary inference data.
            del pixel_values
            del outputs
            del embedding
            gc.collect()
