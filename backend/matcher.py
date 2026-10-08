from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class JewelleryMatcher:

    def __init__(
        self,
        model_loader,
    ):

        self.model_loader = model_loader

        self.project_root = Path(__file__).resolve().parent.parent

        self.matching_root = self.project_root / "matching_data"

        self.gold_dir = self.matching_root / "gold"

        self.prototype_dir = self.matching_root / "prototypes"

        self.catalogue = {
            "gold": [],
            "prototype": [],
        }

        self.embeddings = {
            "gold": None,
            "prototype": None,
        }

        self._load_catalogue()

    # ---------------------------------------------------------
    # LOAD CATALOGUE
    # ---------------------------------------------------------

    def _get_images(
        self,
        directory,
    ):

        if not directory.exists():

            print(f"[WARNING] Missing directory: " f"{directory}")

            return []

        return sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

    def _load_catalogue(self):

        gold_images = self._get_images(self.gold_dir)

        prototype_images = self._get_images(self.prototype_dir)

        print()
        print("=" * 70)
        print("LOADING JEWELLERY CATALOGUE")
        print("=" * 70)

        print(f"Gold images      : {len(gold_images)}")

        print(f"Prototype images : {len(prototype_images)}")

        self.catalogue["gold"] = gold_images

        self.catalogue["prototype"] = prototype_images

        self._create_embeddings("gold")

        self._create_embeddings("prototype")

        print("=" * 70)

    # ---------------------------------------------------------
    # CREATE CATALOGUE EMBEDDINGS
    # ---------------------------------------------------------

    def _create_embeddings(
        self,
        domain,
    ):

        image_paths = self.catalogue[domain]

        if not image_paths:

            self.embeddings[domain] = np.empty(
                (0, 512),
                dtype=np.float32,
            )

            return

        vectors = []

        print()
        print(f"Creating {domain} embeddings...")

        for image_path in image_paths:

            try:

                image = Image.open(image_path).convert("RGB")

                embedding = self.model_loader.get_embedding(image)

                vectors.append(embedding)

            except Exception as exc:

                print(f"[WARNING] Could not process " f"{image_path}: {exc}")

        if not vectors:

            self.embeddings[domain] = np.empty(
                (0, 512),
                dtype=np.float32,
            )

            return

        matrix = np.vstack(vectors).astype(np.float32)

        self.embeddings[domain] = matrix

        print(f"Created {len(matrix)} " f"{domain} embeddings.")

    # ---------------------------------------------------------
    # MATCH
    # ---------------------------------------------------------

    def find_matches(
        self,
        image,
        source,
        top_k=5,
    ):

        if source not in {
            "gold",
            "prototype",
        }:

            raise ValueError("source must be 'gold' " "or 'prototype'")

        target = "prototype" if source == "gold" else "gold"

        query_embedding = self.model_loader.get_embedding(image)

        catalogue_embeddings = self.embeddings[target]

        catalogue_paths = self.catalogue[target]

        if len(catalogue_paths) == 0:

            return []

        # Embeddings are normalized.
        # Therefore dot product = cosine similarity.

        similarities = catalogue_embeddings @ query_embedding

        ranking = np.argsort(similarities)[::-1]

        ranking = ranking[:top_k]

        results = []

        for index in ranking:

            similarity = float(similarities[index])

            image_path = catalogue_paths[index]

            results.append(
                {
                    "id": image_path.stem,
                    "filename": image_path.name,
                    "domain": target,
                    "similarity": similarity,
                    "similarity_percent": round(
                        max(
                            0.0,
                            min(
                                1.0,
                                similarity,
                            ),
                        )
                        * 100,
                        2,
                    ),
                }
            )

        return results
