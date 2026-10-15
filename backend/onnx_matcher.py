import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .database.mongodb import get_catalogue_collection
from .segmentation import prepare_ring_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2_segmented_experiment.onnx"

MODEL_ID = "segmented_dinov2_experiment:734b7306a8f1cb1e"
EMBEDDING_DIM = 512
IMAGE_SIZE = 224
RESIZE_SIZE = 256

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a segmented PIL image to normalized NCHW float32 input.

    The preprocessing follows the usual DINOv2 ImageNet input pipeline:
    resize the shorter side to 256, center-crop to 224, convert to RGB,
    and normalize using ImageNet mean and standard deviation.
    """
    image = image.convert("RGB")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("The image has invalid dimensions.")

    scale = RESIZE_SIZE / min(width, height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    left = (new_width - IMAGE_SIZE) // 2
    top = (new_height - IMAGE_SIZE) // 2

    image = image.crop(
        (
            left,
            top,
            left + IMAGE_SIZE,
            top + IMAGE_SIZE,
        )
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW -> NCHW
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize one embedding."""
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.size != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-D embedding, " f"received {embedding.size}-D."
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError("The model produced an invalid embedding.")

    norm = float(np.linalg.norm(embedding))

    if norm <= 1e-12:
        raise ValueError("The model produced a zero-length embedding.")

    return embedding / norm


class ONNXJewelleryMatcher:
    """Segmented DINOv2 ONNX image matcher backed by MongoDB."""

    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError("Model 2 ONNX file was not found: " f"{MODEL_PATH}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Render CPU deployment: no CUDA or PyTorch dependency.
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs or not outputs:
            raise RuntimeError("The ONNX model has no input or output.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        self.inference_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.catalogue_cache = {}

        print("[MODEL 2] ONNX model loaded:", MODEL_PATH.name)
        print("[MODEL 2] Input:", self.input_name)
        print("[MODEL 2] Output:", self.output_name)
        print("[MODEL 2] Provider:", self.session.get_providers())
        print("[MODEL 2] Model ID:", MODEL_ID)

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Segment and embed one image using the ONNX model."""
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")

        segmented_image = prepare_ring_image(image)
        input_tensor = preprocess_image(segmented_image)

        with self.inference_lock:
            result = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )

        if not result:
            raise RuntimeError("The ONNX model returned no output.")

        return normalize_embedding(result[0])

    def _get_catalogue_embeddings(self, collection_name: str):
        """
        Retrieve cached Model 2 embeddings from a MongoDB collection.

        Documents must contain:
          experimental_embedding
          experimental_embedding_model
        """
        if collection_name not in ("gold", "prototype"):
            raise ValueError("Collection must be 'gold' or 'prototype'.")

        with self.cache_lock:
            if collection_name in self.catalogue_cache:
                return self.catalogue_cache[collection_name]

        collection = get_catalogue_collection(collection_name)

        query = {
            "experimental_embedding_model": MODEL_ID,
            "experimental_embedding": {"$exists": True},
        }

        projection = {
            "_id": 1,
            "name": 1,
            "product_name": 1,
            "image_gridfs_id": 1,
            "image_url": 1,
            "experimental_embedding": 1,
            "experimental_embedding_model": 1,
        }

        vectors = []
        records = []

        for document in collection.find(query, projection):
            try:
                vector = normalize_embedding(
                    np.asarray(
                        document["experimental_embedding"],
                        dtype=np.float32,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

            vectors.append(vector)

            records.append(
                {
                    "id": str(document["_id"]),
                    "name": (
                        document.get("name")
                        or document.get("product_name")
                        or str(document["_id"])
                    ),
                    "image_gridfs_id": (
                        str(document["image_gridfs_id"])
                        if document.get("image_gridfs_id") is not None
                        else None
                    ),
                    "image_url": document.get("image_url"),
                }
            )

        matrix = (
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        )

        data = (matrix, records)

        with self.cache_lock:
            self.catalogue_cache[collection_name] = data

        print(
            f"[MODEL 2] Loaded {len(records)} embeddings "
            f"from MongoDB collection '{collection_name}'."
        )

        return data

    def clear_catalogue_cache(self):
        """Clear cached vectors after regenerating MongoDB embeddings."""
        with self.cache_lock:
            self.catalogue_cache.clear()

    def find_matches(
        self,
        image: Image.Image,
        source: str = "prototype",
        top_k: int = 5,
    ) -> list:
        """
        Match prototype -> gold or gold -> prototype.

        source='prototype' searches the gold collection.
        source='gold' searches the prototype collection.
        """
        source = (source or "prototype").strip().lower()

        if source not in ("gold", "prototype"):
            raise ValueError("source must be either 'gold' or 'prototype'.")

        target_collection = "prototype" if source == "gold" else "gold"

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

        top_k = max(1, min(top_k, 50))

        query_embedding = self._embed_image(image)

        catalogue_matrix, records = self._get_catalogue_embeddings(target_collection)

        if catalogue_matrix.shape[0] == 0:
            return []

        # All vectors are L2-normalized, so dot product equals cosine similarity.
        scores = catalogue_matrix @ query_embedding

        count = min(top_k, len(scores))
        best_indices = np.argsort(scores)[::-1][:count]

        matches = []

        for index in best_indices:
            record = records[int(index)].copy()
            record["score"] = float(scores[index])
            matches.append(record)

        return matches
