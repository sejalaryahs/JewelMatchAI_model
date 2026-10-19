from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    HEAD_LEARNING_RATE,
    WEIGHT_DECAY,
    TEMPERATURE,
    VAL_RATIO,
    NUM_WORKERS,
    SEED,
    DINO_MODEL_NAME,
    EMBEDDING_DIM,
    PROJECTION_DIM,
)
from augmentations import get_train_transform, get_eval_transform
from dataset import JewelleryDataset
from model import JewelleryDINOv2
from loss import nt_xent_loss

# Separate experimental checkpoint; never export or modify ONNX here.
EXPERIMENT_PATH = MODEL_DIR / "jewellery_dinov2_segmented_experiment.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_domain(image_paths):
    """Split gold and prototype images separately."""

    rng = random.Random(SEED)
    train_indices = []
    val_indices = []

    for domain in ("gold", "prototype"):
        indices = [
            i for i, (_, item_domain) in enumerate(image_paths) if item_domain == domain
        ]

        rng.shuffle(indices)

        if len(indices) < 3:
            raise RuntimeError(
                f"Need at least 3 {domain} images; " f"found {len(indices)}."
            )

        val_count = max(
            1,
            min(
                len(indices) - 2,
                round(len(indices) * VAL_RATIO),
            ),
        )

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def make_balanced_batches(image_paths, batch_size, seed):
    """Build batches with equal numbers from each domain."""

    if batch_size < 4 or batch_size % 2:
        raise ValueError("BATCH_SIZE must be even and at least 4.")

    rng = random.Random(seed)

    gold = [i for i, (_, domain) in enumerate(image_paths) if domain == "gold"]

    prototypes = [
        i for i, (_, domain) in enumerate(image_paths) if domain == "prototype"
    ]

    if not gold or not prototypes:
        raise RuntimeError("Both gold and prototype training images are required.")

    per_domain = batch_size // 2

    batch_count = max(
        1,
        (max(len(gold), len(prototypes)) + per_domain - 1) // per_domain,
    )

    rng.shuffle(gold)
    rng.shuffle(prototypes)

    gold_position = 0
    prototype_position = 0
    batches = []

    for _ in range(batch_count):
        batch = []

        for _ in range(per_domain):
            if gold_position >= len(gold):
                rng.shuffle(gold)
                gold_position = 0

            batch.append(gold[gold_position])
            gold_position += 1

        for _ in range(per_domain):
            if prototype_position >= len(prototypes):
                rng.shuffle(prototypes)
                prototype_position = 0

            batch.append(prototypes[prototype_position])
            prototype_position += 1

        rng.shuffle(batch)
        batches.append(batch)

    return batches


def make_validation_batches(dataset_size, batch_size, seed):
    """Use all validation images and avoid singleton batches."""

    if dataset_size < 2:
        raise RuntimeError("At least two validation images are required.")

    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)

    batches = [indices[i : i + batch_size] for i in range(0, dataset_size, batch_size)]

    # Merge a final singleton into the previous batch.
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches[-1])
        batches.pop()

    if any(len(batch) < 2 for batch in batches):
        raise RuntimeError("Validation batching produced a batch smaller than two.")

    return batches


def run_epoch(model, loader, optimizer=None, description="Training"):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch in tqdm(loader, desc=description):
            view1 = batch["view1"].to(DEVICE)
            view2 = batch["view2"].to(DEVICE)

            if view1.shape[0] < 2:
                raise RuntimeError("Contrastive loss requires batch size >= 2.")

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

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError(f"No valid batches found during {description.lower()}.")

    return total_loss / total_batches


def save_checkpoint(checkpoint):
    """
    Atomically save this run's best checkpoint.

    The initial existence check in main() prevents overwriting
    a checkpoint from an earlier run. During this run, improved
    checkpoints are allowed to replace the previous best.
    """

    temporary_path = EXPERIMENT_PATH.with_suffix(".tmp")

    try:
        # Remove only a leftover temporary file from an interrupted save.
        if temporary_path.exists():
            temporary_path.unlink()

        torch.save(checkpoint, temporary_path)

        # Atomic replacement of this experiment's previous best checkpoint.
        os.replace(temporary_path, EXPERIMENT_PATH)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Protect checkpoints from previous runs.
    if EXPERIMENT_PATH.exists():
        raise FileExistsError(
            f"Experimental checkpoint already exists:\n"
            f"{EXPERIMENT_PATH}\n\n"
            "Back it up or rename it before starting a new run."
        )

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found:\n{DATASET_ROOT}")

    print("=" * 65)
    print("SEGMENTED JEWELLERY DINOv2 EXPERIMENT")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Checkpoint: {EXPERIMENT_PATH}")
    print("Existing ONNX files will not be modified.")

    discovery = JewelleryDataset(
        DATASET_ROOT,
        get_eval_transform(),
    )

    all_paths = discovery.image_paths

    train_indices, val_indices = split_by_domain(all_paths)

    train_paths = [all_paths[i] for i in train_indices]

    val_paths = [all_paths[i] for i in val_indices]

    train_dataset = JewelleryDataset(
        DATASET_ROOT,
        get_train_transform(),
        image_paths=train_paths,
    )

    val_dataset = JewelleryDataset(
        DATASET_ROOT,
        get_eval_transform(),
        image_paths=val_paths,
    )

    print(f"\nTraining images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    model = JewelleryDINOv2(freeze_backbone=True).to(DEVICE)

    # Train the embedding and projection heads only.
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        head_parameters,
        lr=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 25} " f"EPOCH {epoch}/{EPOCHS} " f"{'=' * 25}")

        train_batches = make_balanced_batches(
            train_dataset.image_paths,
            BATCH_SIZE,
            SEED + epoch,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batches,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        val_batches = make_validation_batches(
            len(val_dataset),
            BATCH_SIZE,
            SEED + epoch,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batches,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            description="Training",
        )

        val_loss = run_epoch(
            model,
            val_loader,
            optimizer=None,
            description="Validation",
        )

        print(f"\nTrain loss:      {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "model_name": DINO_MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "projection_dim": PROJECTION_DIM,
                "dataset_root": str(DATASET_ROOT),
                "training_mode": "self_supervised_frozen_backbone",
                "seed": SEED,
            }

            save_checkpoint(checkpoint)

            print("Best experimental checkpoint saved:\n" f"{EXPERIMENT_PATH}")
        else:
            print(
                f"No improvement. Best validation loss remains "
                f"{best_val_loss:.6f} (epoch {best_epoch})."
            )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {EXPERIMENT_PATH}")
    print("No ONNX export or deployment was performed.")


if __name__ == "__main__":
    main()
