
import gc
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .database.mongodb import get_jewellery_collection
from .segmentation import prepare_ring_image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "jewellery_dinov2_segmented_experiment.pth"
)

EXPECTED_EMBEDDING_DIM = 512
EXPERIMENTAL_EMBEDDING_FIELD = "experimental_embedding"
EXPERIMENTAL_MODEL_FIELD = "experimental_embedding_model"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model import JewelleryDINOv2  # noqa: E402


def get_model_identifier(checkpoint_path):
    """Generate the same model identifier used by the embedding script."""
    digest = hashlib.sha256()

    with open(checkpoint_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return f"segmented_dinov2_experiment:{digest.hexdigest()[:16]}"


class ExperimentalJewelleryMatcher:
    """
    Experimental segmented DINOv2 matcher.

    Catalogue embeddings are read from MongoDB. Only the uploaded query
    image is segmented and embedded for each matching request.

    Production ONNX embeddings and the production matcher are untouched.
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, device=None):
        self.checkpoint_path = Path(checkpoint_path)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Experimental checkpoint not found: {self.checkpoint_path}"
            )

        self.model_identifier = get_model_identifier(self.checkpoint_path)

        print("\n" + "=" * 65)
        print("INITIALIZING EXPERIMENTAL DINOv2 MATCHER")
        print("=" * 65)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Experimental model ID: {self.model_identifier}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model = JewelleryDINOv2().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        # In-memory cache of embeddings already read from MongoDB.
        # It avoids repeatedly fetching the same vectors during this run.
        self._catalogue_cache = {}

        print("[INFO] Experimental DINOv2 loaded.")
        print("[INFO] MongoDB experimental embedding cache enabled.")
        print("[INFO] Catalogue images will NOT be re-embedded.")
        print("=" * 65)

    @staticmethod
    def _label(document):
        for key in ("jewellery_id", "id", "sku", "name", "filename"):
            value = document.get(key)
            if value not in (None, ""):
                return str(value)

        return str(document.get("_id", "unknown"))

    def _embed_image(self, image):
        """Embed only the uploaded query image."""
        segmented = None

        try:
            rgb_image = image.convert("RGB")
            segmented = prepare_ring_image(rgb_image)

            tensor = (
                self.transform(segmented.convert("RGB"))
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.inference_mode():
                output = self.model(tensor)

                if isinstance(output, (tuple, list)):
                    embedding = output[0]
                elif isinstance(output, dict):
                    embedding = output.get("embedding")
                    if embedding is None:
                        raise ValueError("Model returned no embedding.")
                else:
                    embedding = output

                embedding = F.normalize(embedding.float(), p=2, dim=1)

            vector = (
                embedding.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if vector.ndim != 1 or vector.shape[0] != EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"Expected a {EXPECTED_EMBEDDING_DIM}-dimensional query "
                    f"embedding, got shape {vector.shape}."
                )

            if not np.isfinite(vector).all():
                raise ValueError("Non-finite query embedding produced.")

            return vector

        finally:
            if segmented is not None:
                try:
                    segmented.close()
                except Exception:
                    pass

    def clear_cache(self, collection=None):
        """Clear cached MongoDB vectors for one collection or both."""
        if collection in ("gold", "prototype"):
            self._catalogue_cache.pop(collection, None)
        else:
            self._catalogue_cache.clear()

        print(
            f"[INFO] Experimental embedding cache cleared: "
            f"{collection or 'all'}"
        )

    @staticmethod
    def _catalogue_signature(documents):
        """
        Detect catalogue record changes and changes to experimental vectors.
        """
        return tuple(
            sorted(
                (
                    str(doc.get("_id", "")),
                    str(doc.get(EXPERIMENTAL_MODEL_FIELD, "")),
                    len(doc.get(EXPERIMENTAL_EMBEDDING_FIELD, []))
                    if isinstance(
                        doc.get(EXPERIMENTAL_EMBEDDING_FIELD), list
                    )
                    else -1,
                )
                for doc in documents
            )
        )

    def _get_catalogue_embeddings(self, target):
        start = time.perf_counter()

        database = get_jewellery_collection().database
        collection = database[target]

        # Fetch only metadata and experimental vectors, not image binaries.
        projection = {
            "_id": 1,
            "jewellery_id": 1,
            "id": 1,
            "sku": 1,
            "name": 1,
            "filename": 1,
            "category": 1,
            "image_gridfs_id": 1,
            EXPERIMENTAL_EMBEDDING_FIELD: 1,
            EXPERIMENTAL_MODEL_FIELD: 1,
        }

        documents = list(collection.find({}, projection))

        if not documents:
            self._catalogue_cache.pop(target, None)
            print(f"[INFO] Collection '{target}' is empty.")
            return []

        signature = self._catalogue_signature(documents)
        cached = self._catalogue_cache.get(target)

        if cached and cached["signature"] == signature:
            print(
                f"[TIMING] {target}: in-memory cache hit "
                f"({len(cached['items'])} vectors), "
                f"{time.perf_counter() - start:.3f}s"
            )
            return cached["items"]

        print(
            f"[INFO] Loading {target} experimental embeddings "
            f"from MongoDB ({len(documents)} records)..."
        )

        items = []
        skipped_missing = 0
        skipped_model = 0
        skipped_invalid = 0

        for document in documents:
            document_id = str(document.get("_id", ""))

            # Never mix experimental embeddings from different checkpoints.
            stored_model = document.get(EXPERIMENTAL_MODEL_FIELD)

            if stored_model != self.model_identifier:
                skipped_model += 1
                continue

            raw_embedding = document.get(EXPERIMENTAL_EMBEDDING_FIELD)

            if raw_embedding is None:
                skipped_missing += 1
                continue

            try:
                embedding = np.asarray(raw_embedding, dtype=np.float32)

                if (
                    embedding.ndim != 1
                    or embedding.shape[0] != EXPECTED_EMBEDDING_DIM
                    or not np.isfinite(embedding).all()
                ):
                    skipped_invalid += 1
                    print(
                        f"[WARN] Invalid experimental vector for "
                        f"{self._label(document)}."
                    )
                    continue

                items.append(
                    {
                        "id": document_id,
                        "jewellery_id": self._label(document),
                        "name": str(
                            document.get("name", self._label(document))
                        ),
                        "category": str(document.get("category", "")),
                        "source": target,
                        "image_gridfs_id": (
                            str(document["image_gridfs_id"])
                            if document.get("image_gridfs_id") is not None
                            else None
                        ),
                        "embedding": embedding,
                    }
                )

            except (TypeError, ValueError, OverflowError) as exc:
                skipped_invalid += 1
                print(
                    f"[WARN] Invalid experimental vector for "
                    f"{self._label(document)}: {exc}"
                )

        self._catalogue_cache[target] = {
            "signature": signature,
            "items": items,
        }

        print(
            f"[INFO] {target}: loaded {len(items)} valid vectors; "
            f"wrong model={skipped_model}, "
            f"missing={skipped_missing}, invalid={skipped_invalid}"
        )
        print(
            f"[TIMING] Loaded {target} vectors in "
            f"{time.perf_counter() - start:.3f}s"
        )

        gc.collect()
        return items

    def find_matches(self, image, source="prototype", top_k=5):
        request_start = time.perf_counter()

        source = str(source).strip().lower()

        if source not in ("gold", "prototype"):
            raise ValueError("Source must be 'gold' or 'prototype'.")

        try:
            top_k = max(1, min(int(top_k), 100))
        except (TypeError, ValueError):
            top_k = 5

        # Gold query searches prototype catalogue, and vice versa.
        target = "prototype" if source == "gold" else "gold"

        t0 = time.perf_counter()
        query_embedding = self._embed_image(image)

        print(
            f"[TIMING] Uploaded image embedding: "
            f"{time.perf_counter() - t0:.2f}s"
        )

        candidates = self._get_catalogue_embeddings(target)

        if not candidates:
            print(
                f"[INFO] No usable experimental embeddings in '{target}'. "
                "Check that the collection has records with the current "
                "experimental model identifier."
            )
            print(
                f"[TIMING] Total experimental request: "
                f"{time.perf_counter() - request_start:.2f}s"
            )
            return []

        t0 = time.perf_counter()

        matrix = np.stack([item["embedding"] for item in candidates])

        if matrix.shape[1] != query_embedding.shape[0]:
            raise ValueError(
                "Query and catalogue embedding dimensions do not match: "
                f"{query_embedding.shape[0]} vs {matrix.shape[1]}"
            )

        # Both query and stored embeddings are normalized.
        scores = matrix @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            item = candidates[int(idx)]
            score = float(np.clip(scores[idx], -1.0, 1.0))

            results.append(
                {
                    "id": item["id"],
                    "jewellery_id": item["jewellery_id"],
                    "name": item["name"],
                    "category": item["category"],
                    "source": target,
                    "score": round(score, 6),
                    "similarity": round(score * 100.0, 2),
                    "image_gridfs_id": item["image_gridfs_id"],
                }
            )

        print(
            f"[TIMING] Vector comparison: "
            f"{time.perf_counter() - t0:.4f}s"
        )
        print(
            f"[TIMING] Total experimental request: "
            f"{time.perf_counter() - request_start:.2f}s"
        )

        return results
