
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import DATASET_ROOT, MODEL_DIR, IMAGE_SIZE
from augmentations import get_eval_transform
from model import JewelleryDINOv2


# ------------------------------------------------------------
# PATHS AND SETTINGS
# ------------------------------------------------------------

CHECKPOINT_PATH = (
    MODEL_DIR / "jewellery_dinov2_segmented_experiment.pth"
)

GOLD_DIR = DATASET_ROOT / "gold"
PROTOTYPE_DIR = DATASET_ROOT / "prototypes"

RESULTS_DIR = MODEL_DIR / "segmented_evaluation"
GOLD_TO_PROTOTYPE_CSV = RESULTS_DIR / "gold_to_prototype.csv"
PROTOTYPE_TO_GOLD_CSV = RESULTS_DIR / "prototype_to_gold.csv"
SUMMARY_CSV = RESULTS_DIR / "similarity_summary.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}


# ------------------------------------------------------------
# FIND IMAGES
# ------------------------------------------------------------

def find_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Image folder not found: {folder}"
        )

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ------------------------------------------------------------
# LOAD MODEL CHECKPOINT
# ------------------------------------------------------------

def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print("=" * 65)
    print("LOADING SEGMENTED JEWELLERY MODEL")
    print("=" * 65)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")

    # The checkpoint contains the experiment's trained heads.
    model = JewelleryDINOv2(
        freeze_backbone=True
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Best training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(
        "Saved validation loss: "
        f"{checkpoint.get('val_loss', 'unknown')}"
    )
    print("Model loaded successfully.")

    return model


# ------------------------------------------------------------
# GENERATE IMAGE EMBEDDINGS
# ------------------------------------------------------------

@torch.inference_mode()
def generate_embeddings(model, image_paths):
    transform = get_eval_transform()
    embeddings = []
    valid_paths = []
    failures = []

    for path in tqdm(image_paths, desc="Embedding images"):
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(DEVICE)

            embedding = model.get_embedding(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)

            embeddings.append(embedding.cpu())
            valid_paths.append(path)

        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"\nWARNING: Could not process {path}: {exc}")

    if not embeddings:
        raise RuntimeError(
            f"No valid images found among {len(image_paths)} files."
        )

    return (
        torch.cat(embeddings, dim=0),
        valid_paths,
        failures,
    )


# ------------------------------------------------------------
# RETRIEVE TOP-K NEAREST IMAGES
# ------------------------------------------------------------

def save_rankings(
    query_paths,
    candidate_paths,
    similarities,
    output_path,
    top_k=5,
):
    actual_k = min(top_k, len(candidate_paths))

    values, indices = torch.topk(
        similarities,
        k=actual_k,
        dim=1,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "query_image",
            "rank",
            "matched_image",
            "cosine_similarity",
            "similarity_percent",
        ])

        for query_index, query_path in enumerate(query_paths):
            for rank in range(actual_k):
                candidate_index = indices[
                    query_index, rank
                ].item()

                score = values[
                    query_index, rank
                ].item()

                writer.writerow([
                    str(query_path),
                    rank + 1,
                    str(candidate_paths[candidate_index]),
                    f"{score:.6f}",
                    f"{score * 100:.2f}",
                ])

    best_scores = values[:, 0]

    return {
        "queries": len(query_paths),
        "candidates": len(candidate_paths),
        "top_k": actual_k,
        "mean_best_similarity": best_scores.mean().item(),
        "median_best_similarity": best_scores.median().item(),
        "minimum_best_similarity": best_scores.min().item(),
        "maximum_best_similarity": best_scores.max().item(),
        "mean_all_similarity": similarities.mean().item(),
    }


# ------------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_paths = find_images(GOLD_DIR)
    prototype_paths = find_images(PROTOTYPE_DIR)

    print("\nDataset images found:")
    print(f"Gold: {len(gold_paths)}")
    print(f"Prototypes: {len(prototype_paths)}")

    if not gold_paths or not prototype_paths:
        raise RuntimeError(
            "Both gold and prototypes folders must contain images."
        )

    model = load_model()

    print("\nGenerating gold embeddings...")
    gold_embeddings, valid_gold, gold_failures = (
        generate_embeddings(model, gold_paths)
    )

    print("\nGenerating prototype embeddings...")
    prototype_embeddings, valid_prototypes, prototype_failures = (
        generate_embeddings(model, prototype_paths)
    )

    # Both embedding sets are L2-normalized, so matrix
    # multiplication gives cosine similarity.
    similarities = (
        gold_embeddings @ prototype_embeddings.T
    )

    # Gold -> Prototype
    gold_to_prototype_summary = save_rankings(
        query_paths=valid_gold,
        candidate_paths=valid_prototypes,
        similarities=similarities,
        output_path=GOLD_TO_PROTOTYPE_CSV,
        top_k=5,
    )

    # Prototype -> Gold
    prototype_to_gold_summary = save_rankings(
        query_paths=valid_prototypes,
        candidate_paths=valid_gold,
        similarities=similarities.T,
        output_path=PROTOTYPE_TO_GOLD_CSV,
        top_k=5,
    )

    summaries = [
        (
            "Gold_to_Prototype",
            gold_to_prototype_summary,
        ),
        (
            "Prototype_to_Gold",
            prototype_to_gold_summary,
        ),
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        fieldnames = [
            "direction",
            "queries",
            "candidates",
            "top_k",
            "mean_best_similarity",
            "median_best_similarity",
            "minimum_best_similarity",
            "maximum_best_similarity",
            "mean_all_similarity",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for direction, summary in summaries:
            writer.writerow({
                "direction": direction,
                **summary,
            })

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)

    for direction, summary in summaries:
        print(f"\n{direction.replace('_', ' ')}")
        print(f"Queries: {summary['queries']}")
        print(f"Candidates per query: {summary['candidates']}")
        print(
            "Mean best cosine similarity: "
            f"{summary['mean_best_similarity']:.4f}"
        )
        print(
            "Median best cosine similarity: "
            f"{summary['median_best_similarity']:.4f}"
        )

    print("\nResults saved to:")
    print(GOLD_TO_PROTOTYPE_CSV)
    print(PROTOTYPE_TO_GOLD_CSV)
    print(SUMMARY_CSV)

    total_failures = (
        len(gold_failures) + len(prototype_failures)
    )
    print(f"\nImages that failed to load: {total_failures}")

    print(
        "\nNOTE: These are similarity rankings, not accuracy scores. "
        "Correct-match labels are required to calculate top-1/top-5 accuracy."
    )


if __name__ == "__main__":
    main()
