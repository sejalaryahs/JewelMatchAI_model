
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from .database.mongodb import get_catalogue_collection
from .segmentation import prepare_ring_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "jewellery_dinov2_segmented_experiment.onnx"
)

# Must match the checkpoint used to generate MongoDB embeddings.
MODEL_ID = "segmented_dinov2_experiment:734b7306a8f1cb1e"
EMBEDDING_DIM = 512

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ONNXJewelleryMatcher:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model 2 ONNX file not found: {MODEL_PATH}"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.lock = threading.Lock()
        self.cache = {}

        print(f"[MODEL 2] ONNX model loaded: {MODEL_PATH.name}")
        print(f"[MODEL 2] Model ID: {MODEL_ID}")

    def embed_image(self, image):
        processed = prepare_ring_image(image)

        try:
            tensor = PREPROCESS(processed).unsqueeze(0).numpy()
        finally:
            if processed is not image:
                processed.close()

        with self.lock:
            output = self.session.run(
                [self.output_name],
                {self.input_name: tensor.astype(np.float32)},
            )[0]

        vector = np.asarray(output, dtype=np.float32).reshape(-1)

        if vector.size != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-D embedding, "
                f"received {vector.size}-D."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError("The model returned invalid embedding values.")

        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError("The model returned a zero embedding.")

        return vector / norm

    def _get_catalogue_embeddings(self, target):
        collection = get_catalogue_collection(target)

        cache_key = (target, MODEL_ID)

        if cache_key in self.cache:
            return self.cache[cache_key]

        projection = {
            "_id": 1,
            "experimental_embedding": 1,
            "experimental_embedding_model": 1,
            "name": 1,
            "image_gridfs_id": 1,
            "image_url": 1,
            "image": 1,
        }

        records = collection.find(
            {
                "experimental_embedding_model": MODEL_ID,
                "experimental_embedding": {"$exists": True},
            },
            projection,
        )

        ids = []
        embeddings = []
        documents = []

        for document in records:
            try:
                vector = np.asarray(
                    document["experimental_embedding"],
                    dtype=np.float32,
                ).reshape(-1)

                if vector.size != EMBEDDING_DIM:
                    continue

                norm = np.linalg.norm(vector)
                if norm <= 1e-12 or not np.all(np.isfinite(vector)):
                    continue

                ids.append(document["_id"])
                embeddings.append(vector / norm)
                documents.append(document)

            except (KeyError, TypeError, ValueError):
                continue

        if not embeddings:
            raise ValueError(
                f"No compatible Model 2 embeddings found in "
                f"the '{target}' collection. "
                f"Expected model ID: {MODEL_ID}"
            )

        matrix = np.stack(embeddings).astype(np.float32)

        result = (ids, matrix, documents)
        self.cache[cache_key] = result

        print(
            f"[MODEL 2] Loaded {len(ids)} embeddings "
            f"from '{target}'."
        )

        return result

    def find_matches(self, image, source="prototype", top_k=5):
        source = str(source).strip().lower()

        if source not in {"gold", "prototype"}:
            raise ValueError("Source must be 'gold' or 'prototype'.")

        target = "prototype" if source == "gold" else "gold"

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

        top_k = max(1, min(top_k, 50))

        query = self.embed_image(image)

        ids, matrix, documents = self._get_catalogue_embeddings(target)

        scores = matrix @ query
        count = min(top_k, len(scores))

        indices = np.argsort(scores)[::-1][:count]

        results = []

        for index in indices:
            document = documents[int(index)]

            results.append({
                "id": str(ids[int(index)]),
                "name": document.get("name", ""),
                "score": float(scores[int(index)]),
                "image_gridfs_id": (
                    str(document["image_gridfs_id"])
                    if document.get("image_gridfs_id") is not None
                    else None
                ),
                "image_url": document.get("image_url"),
            })

        return results
