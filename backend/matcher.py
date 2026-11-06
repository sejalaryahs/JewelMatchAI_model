
from typing import Any

import torch
import torch.nn.functional as F


# ============================================================
# JEWELLERY MATCHER
# ============================================================

class JewelleryMatcher:

    def __init__(
        self,
        embedding_dim: int = 512,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):

        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        print("=" * 60)
        print("JEWELLERY MATCHER")
        print("=" * 60)
        print("Embedding dimension:", embedding_dim)
        print("Top K:", top_k)
        print("Similarity threshold:", similarity_threshold)
        print("=" * 60)

    # ========================================================
    # CONVERT EMBEDDING TO TENSOR
    # ========================================================

    def _to_tensor(
        self,
        embedding: Any,
    ) -> torch.Tensor:

        if isinstance(embedding, torch.Tensor):

            tensor = embedding.detach().float()

        else:

            tensor = torch.as_tensor(
                embedding,
                dtype=torch.float32,
            )

        # ----------------------------------------------------
        # HANDLE 2D EMBEDDING
        # ----------------------------------------------------

        if tensor.ndim == 2:

            if tensor.shape[0] != 1:

                raise ValueError(
                    f"Expected one embedding, received {tuple(tensor.shape)}"
                )

            tensor = tensor.squeeze(0)

        # ----------------------------------------------------
        # VALIDATE DIMENSION
        # ----------------------------------------------------

        if tensor.ndim != 1:

            raise ValueError(
                f"Embedding must be 1D, received {tuple(tensor.shape)}"
            )

        if tensor.numel() != self.embedding_dim:

            raise ValueError(
                f"Expected {self.embedding_dim} dimensions, "
                f"received {tensor.numel()}"
            )

        # ----------------------------------------------------
        # VALIDATE VALUES
        # ----------------------------------------------------

        if not torch.isfinite(tensor).all():

            raise ValueError(
                "Embedding contains NaN or infinite values."
            )

        # ----------------------------------------------------
        # VALIDATE MAGNITUDE
        # ----------------------------------------------------

        norm = torch.linalg.vector_norm(tensor)

        if norm.item() == 0:

            raise ValueError(
                "Embedding has zero magnitude."
            )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        return F.normalize(
            tensor,
            p=2,
            dim=0,
        )

    # ========================================================
    # PREPARE QUERY
    # ========================================================

    def prepare_query(
        self,
        query_embedding: Any,
    ) -> torch.Tensor:

        return self._to_tensor(
            query_embedding
        )

    # ========================================================
    # PREPARE PROTOTYPE EMBEDDINGS
    # ========================================================

    def prepare_prototype_embeddings(
        self,
        prototype_embeddings: list[Any],
    ) -> torch.Tensor:

        if not prototype_embeddings:

            raise ValueError(
                "No prototype embeddings provided."
            )

        embeddings = []

        for index, embedding in enumerate(
            prototype_embeddings
        ):

            try:

                embeddings.append(
                    self._to_tensor(embedding)
                )

            except (
                ValueError,
                TypeError,
                RuntimeError,
            ) as error:

                raise ValueError(
                    f"Invalid embedding at index {index}: {error}"
                ) from error

        return torch.stack(
            embeddings,
            dim=0,
        )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    def calculate_similarity(
        self,
        query_embedding: Any,
        prototype_embeddings: list[Any],
    ) -> torch.Tensor:

        query = self.prepare_query(
            query_embedding
        )

        prototypes = self.prepare_prototype_embeddings(
            prototype_embeddings
        )

        return torch.matmul(
            prototypes,
            query,
        )

    # ========================================================
    # MATCH PROTOTYPES
    # ========================================================

    def match(
        self,
        query_embedding: Any,
        prototypes: list[dict],
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict]:

        if not prototypes:

            return []

        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        if top_k is None:

            top_k = self.top_k

        # ----------------------------------------------------
        # SIMILARITY THRESHOLD
        # ----------------------------------------------------

        if similarity_threshold is None:

            similarity_threshold = self.similarity_threshold

        # ----------------------------------------------------
        # VALIDATE TOP K
        # ----------------------------------------------------

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        # ----------------------------------------------------
        # VALIDATE THRESHOLD
        # ----------------------------------------------------

        if not -1.0 <= similarity_threshold <= 1.0:

            raise ValueError(
                "Similarity threshold must be between -1 and 1."
            )

        # ----------------------------------------------------
        # VALIDATE PROTOTYPES
        # ----------------------------------------------------

        valid_prototypes = []

        embeddings = []

        for prototype in prototypes:

            if not isinstance(prototype, dict):

                continue

            embedding = prototype.get(
                "embedding"
            )

            if embedding is None:

                continue

            try:

                tensor = self._to_tensor(
                    embedding
                )

            except (
                ValueError,
                TypeError,
                RuntimeError,
            ):

                continue

            valid_prototypes.append(
                prototype
            )

            embeddings.append(
                tensor
            )

        # ----------------------------------------------------
        # NO VALID PROTOTYPES
        # ----------------------------------------------------

        if not valid_prototypes:

            return []

        # ----------------------------------------------------
        # QUERY EMBEDDING
        # ----------------------------------------------------

        query = self.prepare_query(
            query_embedding
        )

        # ----------------------------------------------------
        # PROTOTYPE TENSOR
        # ----------------------------------------------------

        prototype_tensor = torch.stack(
            embeddings,
            dim=0,
        )

        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------

        with torch.no_grad():

            similarities = torch.matmul(
                prototype_tensor,
                query,
            )

        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        count = min(
            top_k,
            len(valid_prototypes),
        )

        scores, indices = torch.topk(
            similarities,
            k=count,
            largest=True,
            sorted=True,
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        results = []

        for score, index in zip(
            scores.tolist(),
            indices.tolist(),
        ):

            # -----------------------------------------------
            # SIMILARITY THRESHOLD
            # -----------------------------------------------

            if score < similarity_threshold:

                continue

            # -----------------------------------------------
            # ORIGINAL PROTOTYPE RECORD
            # -----------------------------------------------

            prototype = valid_prototypes[index]

            result = dict(
                prototype
            )

            # -----------------------------------------------
            # SIMILARITY SCORE
            # -----------------------------------------------

            result["similarity"] = float(
                score
            )

            # -----------------------------------------------
            # SIMILARITY PERCENTAGE
            # -----------------------------------------------

            result["similarity_percent"] = round(
                float(score) * 100,
                2,
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # COMPARE TWO EMBEDDINGS
    # ========================================================

    def compare(
        self,
        query_embedding: Any,
        prototype_embedding: Any,
    ) -> float:

        query = self.prepare_query(
            query_embedding
        )

        prototype = self._to_tensor(
            prototype_embedding
        )

        similarity = torch.dot(
            query,
            prototype,
        )

        return float(
            similarity.item()
        )

    # ========================================================
    # MATCH INFORMATION
    # UPDATED FOR SUPABASE
    # ========================================================

    def get_match_info(
        self,
        result: dict,
    ) -> dict:

        return {

            # ------------------------------------------------
            # INTERNAL SUPABASE UUID
            # ------------------------------------------------

            "id": result.get(
                "id"
            ),

            # ------------------------------------------------
            # HUMAN-READABLE DESIGN CODE
            # ------------------------------------------------

            "design_code": result.get(
                "design_code"
            ),

            # ------------------------------------------------
            # JEWELLERY DETAILS
            # ------------------------------------------------

            "name": result.get(
                "name"
            ),

            "collection": result.get(
                "collection"
            ),

            "type": result.get(
                "type"
            ),

            "description": result.get(
                "description"
            ),

            # ------------------------------------------------
            # SUPABASE IMAGE
            # ------------------------------------------------

            "image_path": result.get(
                "image_path"
            ),

            "image_url": result.get(
                "image_url"
            ),

            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            "similarity": result.get(
                "similarity"
            ),

            "similarity_percent": result.get(
                "similarity_percent"
            ),

        }
