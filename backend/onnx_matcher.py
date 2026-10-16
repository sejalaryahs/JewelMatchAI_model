"""
Model 2: Segmented DINOv2 ONNX Jewellery Matcher.

Designed for low-memory Render deployments:
- Lazy ONNX session initialization
- Single-threaded CPU inference
- Reduced ONNX Runtime memory retention
- MongoDB-stored, precomputed catalogue embeddings
- Gold -> Prototype and Prototype -> Gold matching

This file does not modify Model 1.
"""

import gc
import io
import logging
import os
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MODEL_ID = "segmented_dinov2_experiment:734b7306a8f1cb1e"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "jewellery_dinov2_segmented_experiment.onnx",
)

IMAGE_SIZE = 224
EMBEDDING_DIM = 512

# ImageNet normalization used by the DINOv2 preprocessing pipeline.
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)

IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class ONNXJewelleryMatcher:
    """Low-memory ONNX jewellery matcher for Model 2."""

    def __init__(self, model_path=None):
        # Do not initialize ONNX here. This keeps application startup light.
        self.model_path = model_path or MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_height = IMAGE_SIZE
        self.input_width = IMAGE_SIZE

        self._session_lock = threading.Lock()
        self._catalogue_lock = threading.Lock()

        # Cache one target catalogue at a time to limit retained memory.
        self._catalogue_cache_source = None
        self._catalogue_cache_embeddings = None
        self._catalogue_cache_documents = None

        logger.info("[MODEL 2] Matcher created; ONNX loading is deferred.")

    def _ensure_session(self):
        """Create the ONNX session only when inference is first required."""
        if self.session is not None:
            return self.session

        with self._session_lock:
            if self.session is not None:
                return self.session

            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    "Model 2 ONNX model was not found: " f"{self.model_path}"
                )

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1

            # Reduce memory retention on small Render instances.
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.log_severity_level = 3

            logger.info("[MODEL 2] Loading ONNX model on demand.")

            session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            model_inputs = session.get_inputs()
            model_outputs = session.get_outputs()

            if not model_inputs or not model_outputs:
                raise RuntimeError("The ONNX model has no usable inputs or outputs.")

            self.input_name = model_inputs[0].name
            self.output_name = model_outputs[0].name

            # Respect the model's declared input dimensions when available.
            input_shape = model_inputs[0].shape
            if len(input_shape) != 4:
                raise RuntimeError(
                    "Expected an ONNX image input with four dimensions "
                    "[batch, channels, height, width]; got "
                    f"{input_shape!r}"
                )

            height = input_shape[2]
            width = input_shape[3]

            if isinstance(height, int) and height > 0:
                self.input_height = height
            if isinstance(width, int) and width > 0:
                self.input_width = width

            self.session = session

            logger.info(
                "[MODEL 2] ONNX session ready; input=%s, size=%sx%s, " "provider=%s",
                self.input_name,
                self.input_width,
                self.input_height,
                session.get_providers()[0],
            )

            return self.session

    @staticmethod
    def _read_image(image):
        """Convert an uploaded image or image path to an RGB PIL image."""
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")

        if isinstance(image, (bytes, bytearray)):
            return ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")

        if hasattr(image, "read"):
            # Support Flask FileStorage and other file-like objects.
            content = image.read()
            try:
                image.seek(0)
            except (AttributeError, OSError):
                pass

            return ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert(
                "RGB"
            )

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")

        raise TypeError(
            "Unsupported image input. Expected a PIL image, bytes, "
            "file-like object, or filesystem path."
        )

    def _preprocess_image(self, image):
        """
        Prepare an image for the ONNX model.

        Uses RGB resizing, ImageNet normalization, and NCHW layout.
        If your exported ONNX model was trained with a different
        preprocessing pipeline, this function must match that pipeline.
        """
        pil_image = self._read_image(image)

        pil_image = pil_image.resize(
            (self.input_width, self.input_height),
            Image.Resampling.BICUBIC,
        )

        array = np.asarray(pil_image, dtype=np.float32) / 255.0
        array = (array - IMAGE_MEAN) / IMAGE_STD

        # HWC -> CHW -> NCHW
        tensor = np.transpose(array, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        return tensor

    @staticmethod
    def _normalize_embedding(embedding):
        """Flatten and L2-normalize a model embedding."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if embedding.size != EMBEDDING_DIM:
            raise ValueError(
                f"Expected a {EMBEDDING_DIM}-dimensional embedding, "
                f"but received {embedding.size} dimensions."
            )

        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("The model produced an invalid embedding.")

        return embedding / norm

    def _embed_image(self, image):
        """Generate a normalized embedding using the lazy ONNX session."""
        session = self._ensure_session()
        tensor = self._preprocess_image(image)

        outputs = session.run(
            [self.output_name],
            {self.input_name: tensor},
        )

        if not outputs:
            raise RuntimeError("The ONNX model returned no output.")

        embedding = self._normalize_embedding(outputs[0])

        del outputs
        del tensor

        return embedding

    @staticmethod
    def _get_collection(source):
        """Resolve the existing Model 2 MongoDB catalogue collection."""
        from backend.database.mongodb import get_catalogue_collection

        collection = get_catalogue_collection(source)
        if collection is None:
            raise RuntimeError(
                f"MongoDB catalogue collection '{source}' is unavailable."
            )

        return collection

    def _get_catalogue_embeddings(self, source):
        """
        Retrieve precomputed Model 2 embeddings from MongoDB.

        Catalogue documents must contain:
          experimental_embedding_model
          experimental_embedding

        This method does not generate or write catalogue embeddings.
        """
        source = str(source).strip().lower()

        if source not in ("gold", "prototype"):
            raise ValueError("Catalogue source must be 'gold' or 'prototype'.")

        with self._catalogue_lock:
            if (
                self._catalogue_cache_source == source
                and self._catalogue_cache_embeddings is not None
            ):
                return (
                    self._catalogue_cache_embeddings,
                    self._catalogue_cache_documents,
                )

            collection = self._get_collection(source)

            cursor = collection.find(
                {
                    "experimental_embedding_model": MODEL_ID,
                    "experimental_embedding": {"$exists": True},
                },
                {
                    "_id": 1,
                    "name": 1,
                    "category": 1,
                    "image_url": 1,
                    "image_gridfs_id": 1,
                    "experimental_embedding": 1,
                },
            )

            vectors = []
            documents = []

            try:
                for document in cursor:
                    try:
                        vector = self._normalize_embedding(
                            document["experimental_embedding"]
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.warning(
                            "[MODEL 2] Skipping invalid embedding in %s: %s",
                            source,
                            exc,
                        )
                        continue

                    vectors.append(vector)
                    documents.append(document)
            finally:
                cursor.close()

            if not vectors:
                logger.warning(
                    "[MODEL 2] No compatible catalogue embeddings found "
                    "in '%s' for model ID '%s'.",
                    source,
                    MODEL_ID,
                )

                empty_embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

                self._catalogue_cache_source = source
                self._catalogue_cache_embeddings = empty_embeddings
                self._catalogue_cache_documents = []

                return empty_embeddings, []

            matrix = np.ascontiguousarray(
                np.stack(vectors, axis=0),
                dtype=np.float32,
            )

            # Replace the prior catalogue cache instead of retaining both.
            self._catalogue_cache_source = source
            self._catalogue_cache_embeddings = matrix
            self._catalogue_cache_documents = documents

            del vectors
            gc.collect()

            logger.info(
                "[MODEL 2] Loaded %s compatible embeddings from '%s'.",
                len(documents),
                source,
            )

            return matrix, documents

    @staticmethod
    def _document_to_result(document):
        """Build a JSON-friendly match result without a similarity score."""
        result = {}

        for key in (
            "name",
            "category",
            "image_url",
            "image_gridfs_id",
        ):
            value = document.get(key)
            if value is not None:
                result[key] = str(value) if key == "image_gridfs_id" else value

        if "_id" in document:
            result["id"] = str(document["_id"])

        return result

    def find_matches(self, image, source="prototype", top_k=5):
        """
        Match an uploaded jewellery image against the opposite catalogue.

        source='gold'      -> search prototype catalogue
        source='prototype' -> search gold catalogue

        Similarity scores are kept internal and are not included in results.
        """
        source = str(source).strip().lower()

        if source not in ("gold", "prototype"):
            raise ValueError("Source must be either 'gold' or 'prototype'.")

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

        top_k = max(1, min(top_k, 50))
        target = "prototype" if source == "gold" else "gold"

        query_embedding = self._embed_image(image)

        catalogue_embeddings, documents = self._get_catalogue_embeddings(target)

        if not documents or catalogue_embeddings.shape[0] == 0:
            return []

        # Both query and catalogue vectors are L2-normalized.
        # Their dot product is cosine similarity.
        similarities = catalogue_embeddings @ query_embedding

        count = min(top_k, len(documents))
        if count == len(documents):
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -count)[-count:]
            indices = indices[np.argsort(similarities[indices])[::-1]]

        results = []
        for index in indices:
            result = self._document_to_result(documents[int(index)])

            # Keep score internal so the existing API can strip it if needed.
            result["score"] = float(similarities[int(index)])
            results.append(result)

        del similarities
        del query_embedding
        gc.collect()

        return results
