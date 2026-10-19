from pathlib import Path

import numpy as np

import torch

from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

from model import JewelleryDINOv2

from augmentations import get_eval_transform

from config import (
    DATASET_ROOT,
    MODEL_PATH,
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# LOAD IMAGES
# ============================================================


def get_images(directory):

    directory = Path(directory)

    if not directory.exists():

        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print()

    print("=" * 80)

    print("DINOv2 JEWELLERY EVALUATION")

    print("=" * 80)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n" f"{MODEL_PATH}\n\n" f"Train the model first."
        )

    model = JewelleryDINOv2(freeze_backbone=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)

    model.eval()

    print()

    print(f"Loaded checkpoint from epoch " f"{checkpoint.get('epoch', '?')}")

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    transform = get_eval_transform()

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    gold_dir = DATASET_ROOT / "gold"

    prototype_dir = DATASET_ROOT / "prototypes"

    gold_images = get_images(gold_dir)

    prototype_images = get_images(prototype_dir)

    print()

    print(f"Gold images: " f"{len(gold_images)}")

    print(f"Prototype images: " f"{len(prototype_images)}")

    if not gold_images:

        raise RuntimeError("No Gold images found.")

    if not prototype_images:

        raise RuntimeError("No Prototype images found.")

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    def create_embeddings(image_paths):

        vectors = []

        valid_paths = []

        for image_path in image_paths:

            try:

                image = Image.open(image_path).convert("RGB")

                tensor = transform(image).unsqueeze(0).to(DEVICE)

                embedding = model.get_embedding(tensor)

                vectors.append(embedding[0].cpu().numpy())

                valid_paths.append(image_path)

            except Exception as exc:

                print(f"[WARNING] " f"Could not process " f"{image_path}: " f"{exc}")

        if not vectors:

            return (
                np.empty(
                    (
                        0,
                        512,
                    ),
                    dtype=np.float32,
                ),
                [],
            )

        return (
            np.vstack(vectors).astype(np.float32),
            valid_paths,
        )

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    print()

    print("Creating Gold embeddings...")

    gold_embeddings, gold_paths = create_embeddings(gold_images)

    print("Creating Prototype embeddings...")

    prototype_embeddings, prototype_paths = create_embeddings(prototype_images)

    # --------------------------------------------------------
    # GOLD → PROTOTYPE
    # --------------------------------------------------------

    print()

    print("=" * 80)

    print("GOLD → PROTOTYPE")

    print("=" * 80)

    similarities = cosine_similarity(
        gold_embeddings,
        prototype_embeddings,
    )

    for index, gold_path in enumerate(gold_paths):

        ranking = np.argsort(similarities[index])[::-1][:5]

        print()

        print(f"Query: " f"{gold_path.name}")

        for rank, prototype_index in enumerate(
            ranking,
            start=1,
        ):

            score = float(
                similarities[
                    index,
                    prototype_index,
                ]
            )

            print(
                f"{rank}. "
                f"{prototype_paths[prototype_index].name}"
                f"  similarity={score:.4f}"
            )

    # --------------------------------------------------------
    # PROTOTYPE → GOLD
    # --------------------------------------------------------

    print()

    print("=" * 80)

    print("PROTOTYPE → GOLD")

    print("=" * 80)

    similarities = cosine_similarity(
        prototype_embeddings,
        gold_embeddings,
    )

    for index, prototype_path in enumerate(prototype_paths):

        ranking = np.argsort(similarities[index])[::-1][:5]

        print()

        print(f"Query: " f"{prototype_path.name}")

        for rank, gold_index in enumerate(
            ranking,
            start=1,
        ):

            score = float(
                similarities[
                    index,
                    gold_index,
                ]
            )

            print(
                f"{rank}. " f"{gold_paths[gold_index].name}" f"  similarity={score:.4f}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
