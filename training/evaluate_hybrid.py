
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import DATASET_ROOT, MODEL_DIR
from hybrid_model import JewelleryHybridModel
from augmentations import get_eval_transform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = MODEL_DIR / "jewellery_hybrid.pth"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TOP_K = 5


def get_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return []

    return sorted(
        [
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def create_embeddings(model, image_paths, transform):
    vectors = []
    valid_paths = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                embedding = model.get_embedding(tensor)

            vectors.append(embedding[0].cpu().numpy())
            valid_paths.append(image_path)

        except Exception as exc:
            print(f"[WARNING] Skipping {image_path.name}: {exc}")

    if not vectors:
        return np.empty((0, 512), dtype=np.float32), []

    return np.vstack(vectors).astype(np.float32), valid_paths


def show_rankings(query_paths, query_embeddings, candidate_paths,
                  candidate_embeddings, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if len(query_paths) == 0 or len(candidate_paths) == 0:
        print("Cannot rank: one or both embedding collections are empty.")
        return

    similarities = cosine_similarity(
        query_embeddings,
        candidate_embeddings,
    )

    for index, query_path in enumerate(query_paths):
        ranking = np.argsort(similarities[index])[::-1][:TOP_K]

        print(f"\nQuery: {query_path.name}")

        for rank, candidate_index in enumerate(ranking, start=1):
            score = float(similarities[index, candidate_index])
            print(
                f"{rank}. {candidate_paths[candidate_index].name}"
                f"  similarity={score:.4f}"
            )


def main():
    print("=" * 80)
    print("HYBRID JEWELLERY MODEL EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid checkpoint not found: {MODEL_PATH}\n"
            "Run training/train_hybrid.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = JewelleryHybridModel(
        freeze_backbone=False,
        pretrained_resnet=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded best checkpoint from epoch: {checkpoint.get('epoch', '?')}")
    print(f"Validation loss: {checkpoint.get('val_loss', float('nan')):.6f}")

    transform = get_eval_transform()

    gold_images = get_images(DATASET_ROOT / "gold")
    prototype_images = get_images(DATASET_ROOT / "prototypes")

    print(f"Gold images found: {len(gold_images)}")
    print(f"Prototype images found: {len(prototype_images)}")

    if not gold_images or not prototype_images:
        raise RuntimeError(
            "Gold or prototype images are missing from processed_segmented."
        )

    print("\nCreating Gold embeddings...")
    gold_embeddings, gold_paths = create_embeddings(
        model, gold_images, transform
    )

    print("Creating Prototype embeddings...")
    prototype_embeddings, prototype_paths = create_embeddings(
        model, prototype_images, transform
    )

    show_rankings(
        gold_paths,
        gold_embeddings,
        prototype_paths,
        prototype_embeddings,
        "GOLD -> PROTOTYPE",
    )

    show_rankings(
        prototype_paths,
        prototype_embeddings,
        gold_paths,
        gold_embeddings,
        "PROTOTYPE -> GOLD",
    )

    print("\n" + "=" * 80)
    print("EVALUATION FINISHED")
    print("=" * 80)
    print(
        "These rankings show embedding similarity, not verified matching "
        "accuracy. Confirm the correct pairs manually before comparing "
        "against the existing DINOv2 matcher."
    )


if __name__ == "__main__":
    main()
