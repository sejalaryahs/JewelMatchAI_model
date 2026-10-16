# import gc
# import threading
# from pathlib import Path

# import numpy as np
# import onnxruntime as ort
# from PIL import Image

# from .database.mongodb import get_catalogue_collection
# from .segmentation import prepare_ring_image

# # ============================================================
# # CONFIGURATION
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2_segmented_experiment.onnx"

# MODEL_ID = "segmented_dinov2_experiment:734b7306a8f1cb1e"

# EMBEDDING_DIM = 512
# IMAGE_SIZE = 224
# RESIZE_SIZE = 256

# # Number of catalogue vectors to retain in memory.
# # Catalogue embeddings are loaded separately for each collection.
# MAX_CACHED_COLLECTIONS = 1

# IMAGENET_MEAN = np.array(
#     [0.485, 0.456, 0.406],
#     dtype=np.float32,
# ).reshape(1, 1, 3)

# IMAGENET_STD = np.array(
#     [0.229, 0.224, 0.225],
#     dtype=np.float32,
# ).reshape(1, 1, 3)


# # ============================================================
# # IMAGE PREPROCESSING
# # ============================================================


# def preprocess_image(image: Image.Image) -> np.ndarray:
#     """
#     Prepare a segmented image for the DINOv2 ONNX model.

#     Steps:
#       1. Convert to RGB.
#       2. Resize the shorter side to 256.
#       3. Center-crop to 224 x 224.
#       4. Normalize using ImageNet statistics.
#       5. Convert HWC to NCHW.

#     Returns:
#         float32 NumPy array with shape (1, 3, 224, 224).
#     """
#     image = image.convert("RGB")

#     width, height = image.size

#     if width <= 0 or height <= 0:
#         raise ValueError("Image has invalid dimensions.")

#     scale = RESIZE_SIZE / min(width, height)

#     new_width = int(round(width * scale))
#     new_height = int(round(height * scale))

#     image = image.resize(
#         (new_width, new_height),
#         Image.Resampling.BICUBIC,
#     )

#     left = (new_width - IMAGE_SIZE) // 2
#     top = (new_height - IMAGE_SIZE) // 2

#     image = image.crop(
#         (
#             left,
#             top,
#             left + IMAGE_SIZE,
#             top + IMAGE_SIZE,
#         )
#     )

#     array = np.asarray(image, dtype=np.float32) / 255.0

#     array = (array - IMAGENET_MEAN) / IMAGENET_STD

#     # HWC -> CHW -> NCHW
#     array = np.transpose(array, (2, 0, 1))
#     array = np.expand_dims(array, axis=0)

#     return np.ascontiguousarray(array, dtype=np.float32)


# # ============================================================
# # EMBEDDING VALIDATION
# # ============================================================


# def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
#     """
#     Validate and L2-normalize an embedding.
#     """
#     embedding = np.asarray(
#         embedding,
#         dtype=np.float32,
#     ).reshape(-1)

#     if embedding.size != EMBEDDING_DIM:
#         raise ValueError(
#             f"Expected {EMBEDDING_DIM}-dimensional embedding; "
#             f"received {embedding.size}."
#         )

#     if not np.all(np.isfinite(embedding)):
#         raise ValueError("Embedding contains invalid numeric values.")

#     norm = float(np.linalg.norm(embedding))

#     if norm <= 1e-12:
#         raise ValueError("Embedding has zero magnitude.")

#     embedding /= norm

#     return embedding


# # ============================================================
# # ONNX JEWELLERY MATCHER
# # ============================================================


# class ONNXJewelleryMatcher:
#     """
#     Segmented DINOv2 ONNX matcher.

#     Prototype query -> gold catalogue
#     Gold query      -> prototype catalogue

#     Model inference uses ONNX Runtime on CPU.
#     No PyTorch or TorchVision is required.
#     """

#     def __init__(self):

#         if not MODEL_PATH.is_file():
#             raise FileNotFoundError(f"Model 2 ONNX file not found: {MODEL_PATH}")

#         # ----------------------------------------------------
#         # LOW-MEMORY ONNX SESSION
#         # ----------------------------------------------------

#         options = ort.SessionOptions()

#         # Avoid additional graph-optimization memory overhead.
#         options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

#         # Use sequential execution with one thread.
#         options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

#         options.intra_op_num_threads = 1
#         options.inter_op_num_threads = 1

#         # Reduce runtime memory overhead.
#         options.enable_mem_pattern = False
#         options.enable_cpu_mem_arena = False

#         self.session = ort.InferenceSession(
#             str(MODEL_PATH),
#             sess_options=options,
#             providers=["CPUExecutionProvider"],
#         )

#         inputs = self.session.get_inputs()
#         outputs = self.session.get_outputs()

#         if not inputs or not outputs:
#             raise RuntimeError("ONNX model has no input or output.")

#         self.input_name = inputs[0].name
#         self.output_name = outputs[0].name

#         # Lock inference to avoid concurrent inference spikes.
#         self.inference_lock = threading.Lock()

#         # Protect catalogue cache operations.
#         self.cache_lock = threading.Lock()

#         # Only one catalogue's vectors are retained at a time.
#         self.cached_collection_name = None
#         self.cached_matrix = None
#         self.cached_records = None

#         print("[MODEL 2] ONNX model loaded:", MODEL_PATH.name)
#         print("[MODEL 2] Model ID:", MODEL_ID)
#         print("[MODEL 2] Input:", self.input_name)
#         print("[MODEL 2] Output:", self.output_name)
#         print(
#             "[MODEL 2] Providers:",
#             self.session.get_providers(),
#         )

#     # ========================================================
#     # EMBED ONE IMAGE
#     # ========================================================

#     def _embed_image(self, image: Image.Image) -> np.ndarray:
#         """
#         Segment and embed a single jewellery image.
#         """
#         if not isinstance(image, Image.Image):
#             raise TypeError("Expected a PIL Image.")

#         # Use the existing GrabCut segmentation.
#         segmented_image = prepare_ring_image(image)

#         input_tensor = preprocess_image(segmented_image)

#         with self.inference_lock:
#             outputs = self.session.run(
#                 [self.output_name],
#                 {
#                     self.input_name: input_tensor,
#                 },
#             )

#         del input_tensor

#         if not outputs:
#             raise RuntimeError("ONNX model returned no output.")

#         embedding = normalize_embedding(outputs[0])

#         del outputs

#         return embedding

#     # ========================================================
#     # LOAD MONGODB CATALOGUE EMBEDDINGS
#     # ========================================================

#     def _get_catalogue_embeddings(self, collection_name: str):
#         """
#         Load the selected catalogue's Model 2 embeddings.

#         Expected MongoDB fields:
#           experimental_embedding
#           experimental_embedding_model

#         The other Model 1 fields are not modified.
#         """
#         if collection_name not in ("gold", "prototype"):
#             raise ValueError("Collection must be 'gold' or 'prototype'.")

#         # Return cached data when the requested collection matches.
#         with self.cache_lock:
#             if self.cached_collection_name == collection_name:
#                 return (
#                     self.cached_matrix,
#                     self.cached_records,
#                 )

#         collection = get_catalogue_collection(collection_name)

#         query = {
#             "experimental_embedding_model": MODEL_ID,
#             "experimental_embedding": {
#                 "$exists": True,
#             },
#         }

#         projection = {
#             "_id": 1,
#             "name": 1,
#             "product_name": 1,
#             "image_gridfs_id": 1,
#             "image_url": 1,
#             "experimental_embedding": 1,
#         }

#         vectors = []
#         records = []

#         cursor = collection.find(
#             query,
#             projection,
#         ).batch_size(100)

#         try:
#             for document in cursor:

#                 try:
#                     vector = normalize_embedding(document["experimental_embedding"])

#                 except (
#                     KeyError,
#                     TypeError,
#                     ValueError,
#                 ):
#                     continue

#                 vectors.append(vector)

#                 gridfs_id = document.get("image_gridfs_id")

#                 records.append(
#                     {
#                         "id": str(document["_id"]),
#                         "name": (
#                             document.get("name")
#                             or document.get("product_name")
#                             or str(document["_id"])
#                         ),
#                         "image_gridfs_id": (
#                             str(gridfs_id) if gridfs_id is not None else None
#                         ),
#                         "image_url": document.get("image_url"),
#                     }
#                 )

#         finally:
#             cursor.close()

#         if vectors:
#             matrix = np.vstack(vectors).astype(
#                 np.float32,
#                 copy=False,
#             )
#         else:
#             matrix = np.empty(
#                 (0, EMBEDDING_DIM),
#                 dtype=np.float32,
#             )

#         del vectors
#         gc.collect()

#         # Cache only one collection to reduce retained memory.
#         with self.cache_lock:

#             self.cached_collection_name = collection_name
#             self.cached_matrix = matrix
#             self.cached_records = records

#         print(
#             f"[MODEL 2] Loaded {len(records)} embeddings " f"from '{collection_name}'."
#         )

#         return matrix, records

#     # ========================================================
#     # CLEAR CATALOGUE CACHE
#     # ========================================================

#     def clear_catalogue_cache(self):
#         """
#         Release cached catalogue vectors.

#         The ONNX session remains loaded.
#         """
#         with self.cache_lock:
#             self.cached_collection_name = None
#             self.cached_matrix = None
#             self.cached_records = None

#         gc.collect()

#         print("[MODEL 2] Catalogue cache cleared.")

#     # ========================================================
#     # FIND MATCHES
#     # ========================================================

#     def find_matches(
#         self,
#         image: Image.Image,
#         source: str = "prototype",
#         top_k: int = 5,
#     ) -> list:
#         """
#         Find visually similar jewellery.

#         source='prototype':
#             Search the gold collection.

#         source='gold':
#             Search the prototype collection.
#         """
#         source = (source or "prototype").strip().lower()

#         if source not in ("gold", "prototype"):
#             raise ValueError("source must be 'gold' or 'prototype'.")

#         target_collection = "prototype" if source == "gold" else "gold"

#         try:
#             top_k = int(top_k)
#         except (TypeError, ValueError):
#             top_k = 5

#         top_k = max(
#             1,
#             min(top_k, 50),
#         )

#         # Embed uploaded query image.
#         query_embedding = self._embed_image(image)

#         # Load the target collection's stored embeddings.
#         matrix, records = self._get_catalogue_embeddings(target_collection)

#         if matrix.shape[0] == 0:
#             return []

#         # Normalized vectors: dot product equals cosine similarity.
#         scores = matrix @ query_embedding

#         result_count = min(
#             top_k,
#             len(scores),
#         )

#         best_indices = np.argsort(scores)[::-1][:result_count]

#         matches = []

#         for index in best_indices:

#             record = records[int(index)].copy()

#             record["score"] = float(scores[int(index)])

#             matches.append(record)

#         del query_embedding
#         del scores

#         return matches
# ################################################################
#####################################################################


import gc
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .database.mongodb import get_catalogue_collection
from .segmentation import prepare_ring_image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2_segmented_experiment.onnx"

MODEL_ID = "segmented_dinov2_experiment:734b7306a8f1cb1e"

EMBEDDING_DIM = 512
IMAGE_SIZE = 224
RESIZE_SIZE = 256
MAX_IMAGE_DIMENSION = 1024

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)

IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a segmented image to a normalized NCHW float32 tensor."""
    rgb_image = image.convert("RGB")

    try:
        width, height = rgb_image.size

        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions.")

        scale = RESIZE_SIZE / min(width, height)

        new_width = max(IMAGE_SIZE, int(round(width * scale)))
        new_height = max(IMAGE_SIZE, int(round(height * scale)))

        resized = rgb_image.resize(
            (new_width, new_height),
            Image.Resampling.BICUBIC,
        )

        try:
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

            try:
                array = np.asarray(cropped, dtype=np.float32)
                array *= 1.0 / 255.0

                array -= IMAGENET_MEAN
                array /= IMAGENET_STD

                # HWC -> NCHW
                array = np.transpose(array, (2, 0, 1))[None, ...]

                return np.ascontiguousarray(array, dtype=np.float32)

            finally:
                cropped.close()

        finally:
            resized.close()

    finally:
        rgb_image.close()


# ============================================================
# EMBEDDING VALIDATION
# ============================================================


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize a 512-dimensional embedding."""
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.size != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dimensional embedding; "
            f"received {embedding.size}."
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError("Embedding contains invalid numeric values.")

    norm = float(np.linalg.norm(embedding))

    if norm <= 1e-12:
        raise ValueError("Embedding has zero magnitude.")

    embedding /= norm
    return embedding


# ============================================================
# ONNX JEWELLERY MATCHER
# ============================================================


class ONNXJewelleryMatcher:
    """
    Segmented DINOv2 ONNX matcher.

    Gold query      -> prototype catalogue
    Prototype query -> gold catalogue

    CPU inference; no PyTorch required.
    """

    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model 2 ONNX file not found: {MODEL_PATH}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs or not outputs:
            raise RuntimeError("ONNX model has no input or output.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        self.inference_lock = threading.Lock()
        self.cache_lock = threading.Lock()

        self.cached_collection_name = None
        self.cached_matrix = None
        self.cached_records = None

        print("[MODEL 2] ONNX model loaded:", MODEL_PATH.name)
        print("[MODEL 2] Model ID:", MODEL_ID)
        print("[MODEL 2] Input:", self.input_name)
        print("[MODEL 2] Output:", self.output_name)
        print("[MODEL 2] Providers:", self.session.get_providers())

    # ========================================================
    # EMBED ONE IMAGE
    # ========================================================

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Segment and embed a single image with bounded input dimensions."""
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")

        working_image = image.convert("RGB")
        segmented_image = None
        input_tensor = None
        outputs = None

        try:
            if min(working_image.size) < 80:
                raise ValueError("Image must be at least 80 x 80 pixels.")

            # Limit segmentation memory for unusually large uploads.
            if max(working_image.size) > MAX_IMAGE_DIMENSION:
                working_image.thumbnail(
                    (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
                    Image.Resampling.LANCZOS,
                )

            segmented_image = prepare_ring_image(working_image)
            input_tensor = preprocess_image(segmented_image)

            with self.inference_lock:
                outputs = self.session.run(
                    [self.output_name],
                    {self.input_name: input_tensor},
                )

            if not outputs:
                raise RuntimeError("ONNX model returned no output.")

            # Return a separate normalized embedding before releasing outputs.
            return normalize_embedding(outputs[0])

        finally:
            del outputs
            del input_tensor

            if segmented_image is not None:
                segmented_image.close()

            working_image.close()
            gc.collect()

    # ========================================================
    # LOAD CATALOGUE EMBEDDINGS
    # ========================================================

    def _get_catalogue_embeddings(self, collection_name: str):
        """Load Model 2 embeddings from the selected MongoDB collection."""
        if collection_name not in ("gold", "prototype"):
            raise ValueError("Collection must be 'gold' or 'prototype'.")

        with self.cache_lock:
            if self.cached_collection_name == collection_name:
                return self.cached_matrix, self.cached_records

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
            }

            vectors = []
            records = []
            cursor = collection.find(query, projection).batch_size(25)

            try:
                for document in cursor:
                    try:
                        vector = normalize_embedding(document["experimental_embedding"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    # Store a compact vector copy for matrix construction.
                    vectors.append(vector.copy())

                    gridfs_id = document.get("image_gridfs_id")

                    records.append(
                        {
                            "id": str(document["_id"]),
                            "name": (
                                document.get("name")
                                or document.get("product_name")
                                or str(document["_id"])
                            ),
                            "image_gridfs_id": (
                                str(gridfs_id) if gridfs_id is not None else None
                            ),
                            "image_url": document.get("image_url"),
                        }
                    )
            finally:
                cursor.close()

            if vectors:
                matrix = np.empty(
                    (len(vectors), EMBEDDING_DIM),
                    dtype=np.float32,
                )

                for index, vector in enumerate(vectors):
                    matrix[index] = vector

                del vectors
            else:
                matrix = np.empty(
                    (0, EMBEDDING_DIM),
                    dtype=np.float32,
                )

            gc.collect()

            self.cached_collection_name = collection_name
            self.cached_matrix = matrix
            self.cached_records = records

            print(
                f"[MODEL 2] Loaded {len(records)} embeddings "
                f"from '{collection_name}'."
            )

            return matrix, records

    # ========================================================
    # CLEAR CATALOGUE CACHE
    # ========================================================

    def clear_catalogue_cache(self):
        """Release cached catalogue embeddings."""
        with self.cache_lock:
            self.cached_collection_name = None
            self.cached_matrix = None
            self.cached_records = None

        gc.collect()
        print("[MODEL 2] Catalogue cache cleared.")

    # ========================================================
    # FIND MATCHES
    # ========================================================

    def find_matches(
        self,
        image: Image.Image,
        source: str = "prototype",
        top_k: int = 5,
    ) -> list:
        """
        source='gold'      -> search prototype catalogue
        source='prototype' -> search gold catalogue
        """
        source = (source or "prototype").strip().lower()

        if source not in ("gold", "prototype"):
            raise ValueError("source must be 'gold' or 'prototype'.")

        target_collection = "prototype" if source == "gold" else "gold"

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

        top_k = max(1, min(top_k, 50))

        query_embedding = None
        scores = None
        best_indices = None

        try:
            query_embedding = self._embed_image(image)

            matrix, records = self._get_catalogue_embeddings(target_collection)

            if matrix.shape[0] == 0:
                return []

            # Both vectors and stored embeddings are L2-normalized.
            scores = matrix @ query_embedding

            result_count = min(top_k, len(scores))

            if result_count < len(scores):
                best_indices = np.argpartition(
                    scores,
                    len(scores) - result_count,
                )[-result_count:]

                best_indices = best_indices[np.argsort(scores[best_indices])[::-1]]
            else:
                best_indices = np.argsort(scores)[::-1]

            matches = []

            for index in best_indices:
                record = records[int(index)].copy()

                # Internal score is retained for ranking but stripped by Flask.
                record["score"] = float(scores[int(index)])
                matches.append(record)

            return matches

        finally:
            del query_embedding
            del scores
            del best_indices
            gc.collect()
