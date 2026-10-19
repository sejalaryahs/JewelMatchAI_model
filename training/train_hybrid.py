from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Support imports when run from the project root.
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (
    DATASET_ROOT,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    HEAD_LEARNING_RATE,
    WEIGHT_DECAY,
    TEMPERATURE,
    VAL_RATIO,
    NUM_WORKERS,
    SEED,
    UNFREEZE_AFTER_EPOCH,
    UNFREEZE_LAST_LAYERS,
)
from dataset import JewelleryDataset
from augmentations import get_train_transform, get_eval_transform
from loss import nt_xent_loss
from hybrid_model import JewelleryHybridModel

# Separate output: never overwrite the existing DINOv2 checkpoint.
HYBRID_MODEL_PATH = MODEL_DIR / "jewellery_hybrid.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_domain_indices(image_paths, val_ratio, seed):
    """Split gold and prototype images independently."""
    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for domain in ("gold", "prototype"):
        indices = [
            i
            for i, (_, image_domain) in enumerate(image_paths)
            if image_domain == domain
        ]

        rng.shuffle(indices)

        if len(indices) < 2:
            train_indices.extend(indices)
            continue

        val_count = max(1, round(len(indices) * val_ratio))
        val_count = min(val_count, len(indices) - 1)

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not train_indices:
        raise RuntimeError("Training split is empty.")

    if len(val_indices) < 2:
        raise RuntimeError(
            "Validation needs at least two images. "
            "Add more images or adjust VAL_RATIO."
        )

    return train_indices, val_indices


def make_optimizer(model):
    """Use a lower learning rate for the DINOv2 backbone."""
    dino_backbone = []
    other_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("dino_model.backbone."):
            dino_backbone.append(parameter)
        else:
            other_parameters.append(parameter)

    groups = []

    if dino_backbone:
        groups.append(
            {
                "params": dino_backbone,
                "lr": LEARNING_RATE,
            }
        )

    if other_parameters:
        groups.append(
            {
                "params": other_parameters,
                "lr": HEAD_LEARNING_RATE,
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY,
    )


def run_epoch(model, loader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch in loader:
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)

            # NT-Xent requires at least two original images per batch.
            if view1.size(0) < 2:
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)

            _, projection1 = model(view1)
            _, projection2 = model(view2)

            loss = nt_xent_loss(
                projection1,
                projection2,
                temperature=TEMPERATURE,
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected. Training stopped.")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

            batch_size = view1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "No valid batches were processed. "
            "Ensure each loader has at least two images per batch."
        )

    return total_loss / total_samples


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("JEWELLERY HYBRID MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Checkpoint: {HYBRID_MODEL_PATH}")

    # Use a deterministic image list for the split.
    discovery_dataset = JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_eval_transform(),
    )

    image_paths = discovery_dataset.image_paths
    train_indices, val_indices = split_domain_indices(
        image_paths,
        VAL_RATIO,
        SEED,
    )

    # Separate dataset instances ensure random augmentations are used
    # for training, while validation uses deterministic preprocessing.
    train_dataset = JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_train_transform(),
        image_paths=image_paths,
    )

    val_dataset = JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_eval_transform(),
        image_paths=image_paths,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=max(2, BATCH_SIZE),
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")

    # ResNet18 starts with ImageNet pretrained weights.
    # The hybrid_model.py architecture is not changed here.
    model = JewelleryHybridModel(
        freeze_backbone=True,
        pretrained_resnet=True,
    ).to(device)

    optimizer = make_optimizer(model)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):

        if epoch == UNFREEZE_AFTER_EPOCH:
            print(f"\nUnfreezing the last " f"{UNFREEZE_LAST_LAYERS} DINOv2 layers.")
            model.unfreeze_last_layers(UNFREEZE_LAST_LAYERS)
            optimizer = make_optimizer(model)

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            training=False,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_name": "JewelleryHybridModel",
                    "embedding_dim": 512,
                    "projection_dim": 256,
                    "dataset_root": str(DATASET_ROOT),
                },
                HYBRID_MODEL_PATH,
            )

            print(f"Saved best hybrid checkpoint: {HYBRID_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("HYBRID TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {HYBRID_MODEL_PATH}")
    print("\nThe existing DINOv2 checkpoint and matcher were not modified.")


if __name__ == "__main__":
    main()
