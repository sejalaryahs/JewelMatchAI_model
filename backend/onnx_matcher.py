"""
Model 2: Segmented DINOv2 ONNX Jewellery Matcher.

Low-memory deployment features:
- Lazy ONNX session initialization
- One reusable CPU inference session
- Single-threaded, serialized inference
- One target-catalogue embedding cache
- Float32 normalized catalogue embeddings
- MongoDB-stored precomputed embeddings
- Gold -> Prototype and Prototype -> Gold matching

Model 1 is not modified.
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

IMAGE_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGE_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


class ONNXJewelleryMatcher:
    """Memory-conscious ONNX matcher for Model 2."""

    def __init__(self, model_path=None):
        # Keep application startup light: load ONNX only on first inference.
        self.model_path = model_path or MODEL_PATH

        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_height = IMAGE_SIZE
        self.input_width = IMAGE_SIZE

        self._session_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._catalogue_lock = threading.Lock()

        # Keep only one target catalogue cached at a time.
        self._catalogue_cache_source = None
        self._catalogue_cache_embeddings = None
        self._catalogue_cache_documents = None

        logger.info("[MODEL 2] Matcher created. ONNX loading is deferred.")

    # --------------------------------------------------------
    # ONNX SESSION
    # --------------------------------------------------------

    def _ensure_session(self):
        """Create and reuse a single ONNX Runtime session."""
        if self.session is not None:
            return self.session

        with self._session_lock:
            if self.session is not None:
                return self.session

            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"Model 2 ONNX model was not found: {self.model_path}"
                )

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1

            # Reduce optional memory retention on small instances.
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.log_severity_level = 3

            logger.info("[MODEL 2] Initializing ONNX Runtime session.")

            session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            model_inputs = session.get_inputs()
            model_outputs = session.get_outputs()

            if not model_inputs or not model_outputs:
                raise RuntimeError("The ONNX model has no usable inputs or outputs.")

            input_meta = model_inputs[0]
            output_meta = model_outputs[0]

            if len(input_meta.shape) != 4:
                raise RuntimeError(
                    "Expected ONNX input shape [batch, channels, height, width]; "
                    f"received {input_meta.shape!r}."
                )

            # Confirm this is a three-channel image model.
            channels = input_meta.shape[1]
            if isinstance(channels, int) and channels != 3:
                raise RuntimeError(f"Expected 3 input channels, received {channels}.")

            height = input_meta.shape[2]
            width = input_meta.shape[3]

            if isinstance(height, int) and height > 0:
                self.input_height = height

            if isinstance(width, int) and width > 0:
                self.input_width = width

            self.input_name = input_meta.name
            self.output_name = output_meta.name
            self.session = session

            logger.info(
                "[MODEL 2] ONNX session ready: input=%s, output=%s, "
                "size=%sx%s, providers=%s",
                self.input_name,
                self.output_name,
                self.input_width,
                self.input_height,
                session.get_providers(),
            )

            return self.session

    # --------------------------------------------------------
    # IMAGE READING
    # --------------------------------------------------------

    @staticmethod
    def _read_image(image):
        """Read an uploaded image, bytes, PIL image, file or path."""
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")

        if isinstance(image, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(image)) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")

        if hasattr(image, "read"):
            content = image.read()

            try:
                image.seek(0)
            except (AttributeError, OSError):
                pass

            with Image.open(io.BytesIO(content)) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")

        raise TypeError(
            "Unsupported image input. Expected a PIL image, bytes, "
            "file-like object, or filesystem path."
        )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    def _preprocess_image(self, image):
        """
        Prepare an image as float32 NCHW input.

        IMPORTANT:
        This matches the current matcher's direct resize and ImageNet
        normalization. It does not perform runtime segmentation or a
        center crop. Change this only if the model's training and
        catalogue-embedding pipeline require different preprocessing.
        """
        pil_image = self._read_image(image)

        try:
            resized = pil_image.resize(
                (self.input_width, self.input_height),
                Image.Resampling.BICUBIC,
            )

            array = np.asarray(resized, dtype=np.float32)
            array *= np.float32(1.0 / 255.0)

            array -= IMAGE_MEAN
            array /= IMAGE_STD

            # HWC -> CHW -> NCHW
            tensor = np.transpose(array, (2, 0, 1))[None, ...]
            tensor = np.ascontiguousarray(tensor, dtype=np.float32)

            del array
            return tensor

        finally:
            pil_image.close()

    # --------------------------------------------------------
    # EMBEDDING NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize_embedding(embedding):
        """Validate and L2-normalize a 512-dimensional vector."""
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size != EMBEDDING_DIM:
            raise ValueError(
                f"Expected a {EMBEDDING_DIM}-dimensional embedding, "
                f"received {vector.size} dimensions."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains NaN or infinite values.")

        norm = float(np.linalg.norm(vector))

        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("The model produced an invalid embedding.")

        # Return a separate, normalized float32 vector.
        return np.asarray(vector / norm, dtype=np.float32)

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    def _embed_image(self, image):
        """Generate a query embedding using the reusable ONNX session."""
        session = self._ensure_session()
        tensor = self._preprocess_image(image)

        try:
            # Serialize inference to avoid overlapping inference peaks
            # when the server receives concurrent requests.
            with self._inference_lock:
                outputs = session.run(
                    [self.output_name],
                    {self.input_name: tensor},
                )

            if not outputs:
                raise RuntimeError("The ONNX model returned no output.")

            raw_embedding = outputs[0]

            # The exported model should produce [batch, 512].
            if raw_embedding.ndim != 2 or raw_embedding.shape[0] != 1:
                raise RuntimeError(
                    "Expected one ONNX embedding with shape [1, 512]; "
                    f"received {raw_embedding.shape!r}."
                )

            return self._normalize_embedding(raw_embedding[0])

        finally:
            del tensor
            if "outputs" in locals():
                del outputs
            gc.collect()

    # --------------------------------------------------------
    # MONGODB COLLECTION
    # --------------------------------------------------------

    @staticmethod
    def _get_collection(source):
        """Resolve the existing Model 2 catalogue collection."""
        from backend.database.mongodb import get_catalogue_collection

        collection = get_catalogue_collection(source)

        if collection is None:
            raise RuntimeError(
                f"MongoDB catalogue collection '{source}' is unavailable."
            )

        return collection

    # --------------------------------------------------------
    # CATALOGUE EMBEDDING CACHE
    # --------------------------------------------------------

    def _get_catalogue_embeddings(self, source):
        """
        Load compatible precomputed embeddings from MongoDB.

        This function does not generate or modify catalogue embeddings.
        Only one target catalogue is retained in the matcher cache.
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

            projection = {
                "_id": 1,
                "name": 1,
                "category": 1,
                "image_url": 1,
                "image_gridfs_id": 1,
                "experimental_embedding": 1,
            }

            cursor = collection.find(
                {
                    "experimental_embedding_model": MODEL_ID,
                    "experimental_embedding": {"$exists": True},
                },
                projection,
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

                    # Keep only fields required by the matching response.
                    result_document = {
                        "_id": document.get("_id"),
                        "name": document.get("name"),
                        "category": document.get("category"),
                        "image_url": document.get("image_url"),
                        "image_gridfs_id": document.get("image_gridfs_id"),
                    }

                    vectors.append(vector)
                    documents.append(result_document)

            finally:
                cursor.close()

            if not vectors:
                logger.warning(
                    "[MODEL 2] No compatible embeddings found in '%s' "
                    "for model ID '%s'.",
                    source,
                    MODEL_ID,
                )

                matrix = np.empty(
                    (0, EMBEDDING_DIM),
                    dtype=np.float32,
                )
                documents = []

            else:
                # Catalogue embeddings are small compared with the model,
                # but avoid retaining both the vector list and matrix.
                matrix = np.empty(
                    (len(vectors), EMBEDDING_DIM),
                    dtype=np.float32,
                )

                for index, vector in enumerate(vectors):
                    matrix[index] = vector

                del vectors
                gc.collect()

            # Replace the previous cached catalogue.
            self._catalogue_cache_source = source
            self._catalogue_cache_embeddings = matrix
            self._catalogue_cache_documents = documents

            logger.info(
                "[MODEL 2] Cached %d compatible embeddings from '%s'.",
                len(documents),
                source,
            )

            return matrix, documents

    # --------------------------------------------------------
    # RESPONSE DOCUMENT
    # --------------------------------------------------------

    @staticmethod
    def _document_to_result(document):
        """Create a JSON-friendly match result."""
        result = {}

        for key in (
            "name",
            "category",
            "image_url",
            "image_gridfs_id",
        ):
            value = document.get(key)

            if value is not None:
                if key == "image_gridfs_id":
                    result[key] = str(value)
                else:
                    result[key] = value

        if document.get("_id") is not None:
            result["id"] = str(document["_id"])

        return result

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    def find_matches(self, image, source="prototype", top_k=5):
        """
        Match an uploaded image against the opposite catalogue.

        source='gold'      -> search prototype catalogue
        source='prototype' -> search gold catalogue

        Internal results retain 'score' for the existing API layer to
        remove as needed. Similarity is not exposed by this class's
        public result formatting.
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
            del query_embedding
            return []

        try:
            # Query and catalogue vectors are normalized, so the dot
            # product is cosine similarity.
            similarities = catalogue_embeddings @ query_embedding

            count = min(top_k, len(documents))

            if count == len(documents):
                indices = np.argsort(similarities)[::-1]
            else:
                indices = np.argpartition(
                    similarities,
                    -count,
                )[-count:]

                indices = indices[np.argsort(similarities[indices])[::-1]]

            results = []

            for index in indices:
                result = self._document_to_result(documents[int(index)])

                # Internal score is preserved for API compatibility.
                result["score"] = float(similarities[int(index)])
                results.append(result)

            return results

        finally:
            del query_embedding

            if "similarities" in locals():
                del similarities

            if "indices" in locals():
                del indices

            gc.collect()
