from typing import Any

import numpy as np

# ============================================================
# JEWELLERY MATCHER
# MODEL 3 - ONNX VERSION - NUMPY
# BIDIRECTIONAL MATCHING
# ============================================================


class JewelleryMatcher:

    def __init__(
        self,
        embedding_dim: int = 512,
        top_k: int = 12,
        similarity_threshold: float = 0.0,
    ):

        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        print("=" * 60)
        print("JEWELLERY MATCHER - ONNX")
        print("=" * 60)
        print("Embedding dimension:", embedding_dim)
        print("Top K:", top_k)
        print("Similarity threshold:", similarity_threshold)
        print("Matching: Bidirectional")
        print("=" * 60)

    # ========================================================
    # CONVERT EMBEDDING TO NUMPY
    # ========================================================

    def _to_numpy(
        self,
        embedding: Any,
    ) -> np.ndarray:

        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()

        try:
            array = np.asarray(
                embedding,
                dtype=np.float32,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unable to convert embedding to NumPy: {error}"
            ) from error

        # Handle a single batch.
        if array.ndim == 2:

            if array.shape[0] != 1:
                raise ValueError(f"Expected one embedding, received {array.shape}")

            array = array.squeeze(0)

        if array.ndim != 1:
            raise ValueError(f"Embedding must be 1D, received {array.shape}")

        if array.size != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} dimensions, " f"received {array.size}"
            )

        if not np.isfinite(array).all():
            raise ValueError("Embedding contains NaN or infinite values.")

        norm = np.linalg.norm(array)

        if norm <= 1e-12:
            raise ValueError("Embedding has zero magnitude.")

        return (array / norm).astype(np.float32)

    # ========================================================
    # PREPARE QUERY
    # ========================================================

    def prepare_query(
        self,
        query_embedding: Any,
    ) -> np.ndarray:

        return self._to_numpy(query_embedding)

    # ========================================================
    # PREPARE CATALOGUE EMBEDDINGS
    # ========================================================

    def prepare_prototype_embeddings(
        self,
        prototype_embeddings: list[Any],
    ) -> np.ndarray:

        if not prototype_embeddings:
            raise ValueError("No catalogue embeddings provided.")

        embeddings = []

        for index, embedding in enumerate(prototype_embeddings):

            try:
                embeddings.append(self._to_numpy(embedding))

            except (
                ValueError,
                TypeError,
                RuntimeError,
            ) as error:

                raise ValueError(
                    f"Invalid embedding at index {index}: {error}"
                ) from error

        return np.stack(
            embeddings,
            axis=0,
        )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    def calculate_similarity(
        self,
        query_embedding: Any,
        prototype_embeddings: list[Any],
    ) -> np.ndarray:

        query = self.prepare_query(query_embedding)

        prototypes = self.prepare_prototype_embeddings(prototype_embeddings)

        return np.matmul(
            prototypes,
            query,
        )

    # ========================================================
    # MATCH CATALOGUE ITEMS
    # ========================================================

    def match(
        self,
        query_embedding: Any,
        prototypes: list[dict],
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict]:

        # ----------------------------------------------------
        # EMPTY CATALOGUE
        # ----------------------------------------------------

        if not prototypes:
            return []

        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        if top_k is None:
            top_k = self.top_k

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # ----------------------------------------------------
        # SIMILARITY THRESHOLD
        # ----------------------------------------------------

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between -1 and 1.")

        # ----------------------------------------------------
        # PREPARE QUERY
        # ----------------------------------------------------

        query = self.prepare_query(query_embedding)

        # ----------------------------------------------------
        # PREPARE VALID CATALOGUE ITEMS
        # ----------------------------------------------------

        valid_items = []
        embeddings = []

        for item in prototypes:

            if not isinstance(item, dict):
                continue

            embedding = item.get("embedding")

            if embedding is None:
                continue

            try:
                normalized_embedding = self._to_numpy(embedding)

            except (
                ValueError,
                TypeError,
                RuntimeError,
            ):
                continue

            valid_items.append(item)
            embeddings.append(normalized_embedding)

        if not valid_items:
            return []

        # ----------------------------------------------------
        # STACK EMBEDDINGS
        # ----------------------------------------------------

        catalogue_embeddings = np.stack(
            embeddings,
            axis=0,
        )

        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------

        similarities = np.matmul(
            catalogue_embeddings,
            query,
        )

        # ----------------------------------------------------
        # SORT BY SIMILARITY
        # ----------------------------------------------------

        sorted_indices = np.argsort(similarities)[::-1]

        # ----------------------------------------------------
        # RETURN TOP 12 VALID RESULTS
        # ----------------------------------------------------

        results = []

        for index in sorted_indices:

            score = float(similarities[index])

            if score < similarity_threshold:
                continue

            item = dict(valid_items[index])

            item["similarity"] = score

            item["similarity_percent"] = round(
                score * 100,
                2,
            )

            results.append(item)

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # COMPARE TWO EMBEDDINGS
    # ========================================================

    def compare(
        self,
        query_embedding: Any,
        prototype_embedding: Any,
    ) -> float:

        query = self.prepare_query(query_embedding)

        prototype = self._to_numpy(prototype_embedding)

        return float(
            np.dot(
                query,
                prototype,
            )
        )

    # ========================================================
    # FORMAT MATCH INFORMATION
    # ========================================================

    def get_match_info(
        self,
        result: dict,
    ) -> dict:

        return {
            "id": result.get("id"),
            "design_code": result.get("design_code"),
            "name": result.get("name"),
            "collection": result.get("collection"),
            "type": result.get("type"),
            "description": result.get("description"),
            "image_path": result.get("image_path"),
            "image_url": result.get("image_url"),
            "similarity": result.get("similarity"),
            "similarity_percent": result.get("similarity_percent"),
        }


# ============================================================
# TEST MATCHER
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing Jewellery Matcher")

    matcher = JewelleryMatcher(
        embedding_dim=512,
        top_k=12,
        similarity_threshold=0.0,
    )

    # Create sample query embedding.
    query = np.random.rand(512).astype(np.float32)

    query = query / np.linalg.norm(query)

    # Create 20 sample catalogue embeddings.
    catalogue = []

    for index in range(20):

        embedding = np.random.rand(512).astype(np.float32)

        embedding = embedding / np.linalg.norm(embedding)

        catalogue.append(
            {
                "id": str(index + 1),
                "design_code": f"DESIGN-{index + 1}",
                "type": "prototype",
                "embedding": embedding,
            }
        )

    # Test matching.
    results = matcher.match(
        query_embedding=query,
        prototypes=catalogue,
        top_k=12,
    )

    print()
    print("=" * 60)
    print("MATCHING TEST RESULTS")
    print("=" * 60)

    print("Requested results:", 12)
    print("Returned results:", len(results))

    for result in results:

        print(
            result["design_code"],
            result["similarity_percent"],
        )

    print("=" * 60)
