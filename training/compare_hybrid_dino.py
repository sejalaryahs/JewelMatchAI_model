from pathlib import Path
import csv
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Make imports work when running this file from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(TRAINING_DIR))

from model_loader import ModelLoader
from hybrid_model import JewelleryHybridModel
from config import IMAGE_SIZE, EMBEDDING_DIM

DATASET_ROOT = PROJECT_ROOT / "processed_segmented"
HYBRID_CHECKPOINT = PROJECT_ROOT / "models" / "jewellery_hybrid.pth"
OUTPUT_CSV = PROJECT_ROOT / "models" / "hybrid_vs_dino_comparison.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = torch.device("cpu")

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.files = sorted(
            path
            for path in self.folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            tensor = EVAL_TRANSFORM(image.convert("RGB"))

        return tensor, str(path)


def load_hybrid_model():
    if not HYBRID_CHECKPOINT.exists():
        raise FileNotFoundError(f"Hybrid checkpoint not found: {HYBRID_CHECKPOINT}")

    checkpoint = torch.load(
        HYBRID_CHECKPOINT,
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

    print(
        "Hybrid checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print(
        "Hybrid validation loss:",
        checkpoint.get("val_loss", "unknown"),
    )

    return model


def collect_paths():
    gold = ImageDataset(DATASET_ROOT / "gold")
    prototypes = ImageDataset(DATASET_ROOT / "prototypes")

    if not gold.files or not prototypes.files:
        raise RuntimeError(
            "Gold or prototype images were not found under " f"{DATASET_ROOT}"
        )

    print(f"Gold images: {len(gold)}")
    print(f"Prototype images: {len(prototypes)}")

    return gold, prototypes


@torch.inference_mode()
def get_hybrid_embeddings(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    embeddings = []
    paths = []

    for batch, batch_paths in loader:
        batch = batch.to(DEVICE)
        vectors = model.get_embedding(batch)
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)

        embeddings.append(vectors.cpu().numpy())
        paths.extend(batch_paths)

    return np.concatenate(embeddings), paths


def get_onnx_embeddings(loader, dataset):
    embeddings = []
    paths = []

    for path in dataset.files:
        with Image.open(path) as image:
            # Pass PIL RGB image to the existing production preprocessing.
            vector = loader.get_embedding(image.convert("RGB"))

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)

        if vector.size != EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected ONNX embedding dimension for {path}: " f"{vector.size}"
            )

        norm = np.linalg.norm(vector)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError(f"Invalid ONNX embedding for {path}")

        embeddings.append(vector / norm)
        paths.append(str(path))

    return np.vstack(embeddings), paths


def compare_direction(
    source_name,
    source_paths,
    source_dino,
    source_hybrid,
    target_name,
    target_paths,
    target_dino,
    target_hybrid,
    output_rows,
):
    print(f"\n{'=' * 75}")
    print(f"{source_name.upper()} -> {target_name.upper()}")
    print(f"{'=' * 75}")

    dino_scores = source_dino @ target_dino.T
    hybrid_scores = source_hybrid @ target_hybrid.T

    for i, source_path in enumerate(source_paths):
        dino_top = np.argsort(dino_scores[i])[::-1][:5]
        hybrid_top = np.argsort(hybrid_scores[i])[::-1][:5]

        print(f"\nQuery: {Path(source_path).name}")
        print("  DINOv2 ONNX top 5:")
        for rank, j in enumerate(dino_top, start=1):
            print(
                f"    {rank}. {Path(target_paths[j]).name}"
                f" | cosine={dino_scores[i, j]:.4f}"
            )

        print("  Hybrid top 5:")
        for rank, j in enumerate(hybrid_top, start=1):
            print(
                f"    {rank}. {Path(target_paths[j]).name}"
                f" | cosine={hybrid_scores[i, j]:.4f}"
            )

        dino_rank = {
            int(index): rank
            for rank, index in enumerate(
                np.argsort(dino_scores[i])[::-1],
                start=1,
            )
        }
        hybrid_rank = {
            int(index): rank
            for rank, index in enumerate(
                np.argsort(hybrid_scores[i])[::-1],
                start=1,
            )
        }

        for j, target_path in enumerate(target_paths):
            output_rows.append(
                {
                    "direction": f"{source_name}_to_{target_name}",
                    "query_image": Path(source_path).name,
                    "candidate_image": Path(target_path).name,
                    "dino_cosine": round(float(dino_scores[i, j]), 6),
                    "dino_rank": dino_rank[j],
                    "hybrid_cosine": round(float(hybrid_scores[i, j]), 6),
                    "hybrid_rank": hybrid_rank[j],
                    "hybrid_rank_change": dino_rank[j] - hybrid_rank[j],
                }
            )


def main():
    print("=" * 75)
    print("HYBRID MODEL vs EXISTING DINOv2 ONNX")
    print("Evaluation only; production files remain unchanged.")
    print("=" * 75)

    gold_dataset, prototype_dataset = collect_paths()

    print("\nLoading existing ONNX baseline...")
    onnx_loader = ModelLoader()

    print("\nLoading trained hybrid checkpoint...")
    hybrid_model = load_hybrid_model()

    print("\nGenerating ONNX embeddings...")
    gold_dino, gold_paths = get_onnx_embeddings(onnx_loader, gold_dataset)
    prototype_dino, prototype_paths = get_onnx_embeddings(
        onnx_loader, prototype_dataset
    )

    print("\nGenerating hybrid embeddings...")
    gold_hybrid, _ = get_hybrid_embeddings(hybrid_model, gold_dataset)
    prototype_hybrid, _ = get_hybrid_embeddings(hybrid_model, prototype_dataset)

    if gold_dino.shape[1] != EMBEDDING_DIM:
        raise ValueError("Unexpected gold DINO embedding dimension.")
    if prototype_dino.shape[1] != EMBEDDING_DIM:
        raise ValueError("Unexpected prototype DINO embedding dimension.")
    if gold_hybrid.shape[1] != EMBEDDING_DIM:
        raise ValueError("Unexpected gold hybrid embedding dimension.")
    if prototype_hybrid.shape[1] != EMBEDDING_DIM:
        raise ValueError("Unexpected prototype hybrid embedding dimension.")

    output_rows = []

    compare_direction(
        "gold",
        gold_paths,
        gold_dino,
        gold_hybrid,
        "prototype",
        prototype_paths,
        prototype_dino,
        prototype_hybrid,
        output_rows,
    )

    compare_direction(
        "prototype",
        prototype_paths,
        prototype_dino,
        prototype_hybrid,
        "gold",
        gold_paths,
        gold_hybrid,
        gold_dino,
        output_rows,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "direction",
                "query_image",
                "candidate_image",
                "dino_cosine",
                "dino_rank",
                "hybrid_cosine",
                "hybrid_rank",
                "hybrid_rank_change",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nComparison CSV saved to:\n{OUTPUT_CSV}")
    print("\nComparison completed. No production files were changed.")


if __name__ == "__main__":
    main()
