# ============================================================
# JEWELLERY DINOv2 TRAINING
# ============================================================
#
# Self-supervised / contrastive jewellery representation
# learning.
#
# IMPORTANT:
#   Gold and Prototype images are NOT artificially paired.
#
# Each image creates two augmented views:
#
#       Image
#        / \
#       /   \
#   View 1  View 2
#
# View 1 and View 2 are the positive pair.
#
# Gold and Prototype are balanced at the BATCH level so that
# the larger Prototype dataset does not dominate training.
#
# ============================================================

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    MODEL_DIR,
    MODEL_PATH,
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

from augmentations import (
    get_train_transform,
    get_eval_transform,
)

from dataset import JewelleryDataset

from model import JewelleryDINOv2

from loss import nt_xent_loss

# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# BALANCED BATCH SAMPLER
# ============================================================


class BalancedBatchSampler:
    """
    Produces batches containing approximately:

        50% Gold
        50% Prototype

    Example with BATCH_SIZE = 4:

        [Gold, Gold, Prototype, Prototype]

    Important:
        This does NOT create Gold/Prototype pairs.

    It only controls how frequently each domain appears
    during training.

    The smaller Gold dataset is cycled through multiple times.
    """

    def __init__(
        self,
        gold_indices,
        prototype_indices,
        batch_size=4,
        seed=42,
    ):

        if batch_size < 2:

            raise ValueError("BATCH_SIZE must be at least 2.")

        if batch_size % 2 != 0:

            raise ValueError(
                "BATCH_SIZE must be even for balanced " "Gold/Prototype batches."
            )

        if not gold_indices:

            raise ValueError("No Gold training images available.")

        if not prototype_indices:

            raise ValueError("No Prototype training images available.")

        self.gold_indices = list(gold_indices)

        self.prototype_indices = list(prototype_indices)

        self.batch_size = batch_size

        self.gold_per_batch = batch_size // 2

        self.prototype_per_batch = batch_size // 2

        self.seed = seed

        # ----------------------------------------------------
        # Number of batches is based on the larger domain.
        #
        # With:
        #   32 Gold
        #   162 Prototype
        #
        # We get:
        #   162 / 2 = 81 batches
        #
        # Gold gets reused.
        # ----------------------------------------------------

        self.num_batches = (
            max(len(self.gold_indices), len(self.prototype_indices))
            // self.prototype_per_batch
        )

        if self.num_batches < 1:

            self.num_batches = 1

    def __iter__(self):

        rng = random.Random(self.seed)

        # ----------------------------------------------------
        # Shuffle each domain independently.
        # ----------------------------------------------------

        gold = self.gold_indices.copy()

        prototype = self.prototype_indices.copy()

        rng.shuffle(gold)

        rng.shuffle(prototype)

        gold_position = 0

        prototype_position = 0

        # ----------------------------------------------------
        # Build batches.
        # ----------------------------------------------------

        for _ in range(self.num_batches):

            batch = []

            # ------------------------------------------------
            # Gold samples
            # ------------------------------------------------

            for _ in range(self.gold_per_batch):

                if gold_position >= len(gold):

                    gold = self.gold_indices.copy()

                    rng.shuffle(gold)

                    gold_position = 0

                batch.append(gold[gold_position])

                gold_position += 1

            # ------------------------------------------------
            # Prototype samples
            # ------------------------------------------------

            for _ in range(self.prototype_per_batch):

                if prototype_position >= len(prototype):

                    prototype = self.prototype_indices.copy()

                    rng.shuffle(prototype)

                    prototype_position = 0

                batch.append(prototype[prototype_position])

                prototype_position += 1

            # ------------------------------------------------
            # Shuffle the order inside the batch.
            #
            # This prevents the model from seeing:
            #
            # Gold Gold Prototype Prototype
            #
            # in the exact same order every time.
            # ------------------------------------------------

            rng.shuffle(batch)

            yield batch

    def __len__(self):

        return self.num_batches


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================


def create_stratified_split(
    image_paths,
    val_ratio,
    seed,
):
    """
    Creates separate Gold and Prototype train/validation
    splits.

    This prevents one domain from disappearing from validation.
    """

    rng = random.Random(seed)

    gold_indices = [
        index for index, (_, domain) in enumerate(image_paths) if domain == "gold"
    ]

    prototype_indices = [
        index for index, (_, domain) in enumerate(image_paths) if domain == "prototype"
    ]

    rng.shuffle(gold_indices)

    rng.shuffle(prototype_indices)

    # --------------------------------------------------------
    # Validation count.
    # --------------------------------------------------------

    gold_val_count = max(1, int(len(gold_indices) * val_ratio))

    prototype_val_count = max(1, int(len(prototype_indices) * val_ratio))

    gold_val = gold_indices[:gold_val_count]

    gold_train = gold_indices[gold_val_count:]

    prototype_val = prototype_indices[:prototype_val_count]

    prototype_train = prototype_indices[prototype_val_count:]

    train_indices = gold_train + prototype_train

    val_indices = gold_val + prototype_val

    rng.shuffle(train_indices)

    rng.shuffle(val_indices)

    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(f"Gold training:       {len(gold_train)}")

    print(f"Gold validation:     {len(gold_val)}")

    print(f"Prototype training:  {len(prototype_train)}")

    print(f"Prototype validation:{len(prototype_val)}")

    print(f"Total training:      {len(train_indices)}")

    print(f"Total validation:    {len(val_indices)}")

    return (
        train_indices,
        val_indices,
        gold_train,
        prototype_train,
    )


# ============================================================
# CREATE TRAINING DATASET
# ============================================================


def create_training_dataset():

    return JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_train_transform(),
    )


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================


def create_validation_dataset():

    return JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_eval_transform(),
    )


# ============================================================
# CREATE TRAINING LOADER
# ============================================================


def create_training_loader(
    dataset,
    gold_indices,
    prototype_indices,
):

    batch_sampler = BalancedBatchSampler(
        gold_indices=gold_indices,
        prototype_indices=prototype_indices,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# ============================================================
# CREATE VALIDATION LOADER
# ============================================================


def create_validation_loader(
    dataset,
    val_indices,
):

    sampler = SubsetRandomSampler(val_indices)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader


# ============================================================
# TRAIN ONE EPOCH
# ============================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
):

    model.train()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Training Epoch {epoch}",
        leave=True,
    )

    for batch in progress:

        view1 = batch["view1"].to(DEVICE, non_blocking=True)

        view2 = batch["view2"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, projection1 = model(view1)

        _, projection2 = model(view2)

        loss = nt_xent_loss(
            projection1,
            projection2,
            temperature=TEMPERATURE,
        )

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping.
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if len(loader) == 0:

        return 0.0

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=True,
    )

    for batch in progress:

        view1 = batch["view1"].to(DEVICE, non_blocking=True)

        view2 = batch["view2"].to(DEVICE, non_blocking=True)

        _, projection1 = model(view1)

        _, projection2 = model(view2)

        loss = nt_xent_loss(
            projection1,
            projection2,
            temperature=TEMPERATURE,
        )

        total_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if len(loader) == 0:

        return 0.0

    return total_loss / len(loader)


# ============================================================
# CREATE OPTIMIZER
# ============================================================


def create_optimizer(model):

    # --------------------------------------------------------
    # Initially only embedding/projection heads are trainable.
    # --------------------------------------------------------

    backbone_parameters = []

    head_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:

            continue

        if name.startswith("backbone."):

            backbone_parameters.append(parameter)

        else:

            head_parameters.append(parameter)

    parameter_groups = []

    if head_parameters:

        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": HEAD_LEARNING_RATE,
            }
        )

    if backbone_parameters:

        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": LEARNING_RATE,
            }
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# ============================================================
# UPDATE BACKBONE TRAINING
# ============================================================


def unfreeze_backbone_if_needed(
    model,
    epoch,
):

    if epoch != UNFREEZE_AFTER_EPOCH:

        return False

    print("\n" + "=" * 70)
    print("UNFREEZING LAST DINOv2 LAYERS")
    print("=" * 70)

    print(f"Unfreezing last " f"{UNFREEZE_LAST_LAYERS} layers.")

    model.unfreeze_last_layers(UNFREEZE_LAST_LAYERS)

    return True


# ============================================================
# MAIN
# ============================================================


def main():

    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JEWELLERY DINOv2 TRAINING")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    print(f"Dataset: {DATASET_ROOT}")

    print(f"Batch size: {BATCH_SIZE}")

    print(f"Epochs: {EPOCHS}")

    print(f"Temperature: {TEMPERATURE}")

    # --------------------------------------------------------
    # Check dataset.
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nDataset directory does not exist:\n" f"{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # Create datasets.
    #
    # We need the paths first so that train/validation can
    # use separate transformations.
    # --------------------------------------------------------

    discovery_dataset = JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_eval_transform(),
    )

    all_image_paths = discovery_dataset.image_paths

    (
        train_indices,
        val_indices,
        gold_train,
        prototype_train,
    ) = create_stratified_split(
        image_paths=all_image_paths,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_train_transform(),
        image_paths=[all_image_paths[index] for index in train_indices],
    )

    # --------------------------------------------------------
    # Because train_dataset contains only train images,
    # rebuild local Gold/Prototype indices.
    # --------------------------------------------------------

    local_gold_indices = [
        index
        for index, (_, domain) in enumerate(train_dataset.image_paths)
        if domain == "gold"
    ]

    local_prototype_indices = [
        index
        for index, (_, domain) in enumerate(train_dataset.image_paths)
        if domain == "prototype"
    ]

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = JewelleryDataset(
        root_dir=DATASET_ROOT,
        transform=get_eval_transform(),
        image_paths=[all_image_paths[index] for index in val_indices],
    )

    # --------------------------------------------------------
    # Training loader
    # --------------------------------------------------------

    train_loader = create_training_loader(
        dataset=train_dataset,
        gold_indices=local_gold_indices,
        prototype_indices=local_prototype_indices,
    )

    # --------------------------------------------------------
    # Validation loader
    # --------------------------------------------------------

    val_loader = create_validation_loader(
        dataset=val_dataset,
        val_indices=list(range(len(val_dataset))),
    )

    print("\n" + "=" * 70)
    print("BALANCED TRAINING")
    print("=" * 70)

    print("Each training batch contains approximately:")

    print(f"  {BATCH_SIZE // 2} Gold")

    print(f"  {BATCH_SIZE // 2} Prototype")

    print("\nNo Gold/Prototype artificial pairing is used.")

    print("\nTraining batches per epoch:" f" {len(train_loader)}")

    print("Validation batches:" f" {len(val_loader)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INITIALIZING DINOv2")
    print("=" * 70)

    model = JewelleryDINOv2(freeze_backbone=True)

    model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(model)

    # --------------------------------------------------------
    # Best validation loss.
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print("\n")
        print("=" * 70)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 70)

        # ----------------------------------------------------
        # Unfreeze last DINOv2 layers at configured epoch.
        # ----------------------------------------------------

        if unfreeze_backbone_if_needed(
            model,
            epoch,
        ):

            optimizer = create_optimizer(model)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        val_loss = validate(
            model=model,
            loader=val_loader,
        )

        print("\n" + "-" * 70)

        print(f"Epoch {epoch}/{EPOCHS}")

        print(f"Train Loss: {train_loss:.6f}")

        print(f"Val Loss:   {val_loss:.6f}")

        print("-" * 70)

        # ----------------------------------------------------
        # Save best checkpoint.
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": "facebook/dinov2-base",
                "embedding_dim": 512,
                "projection_dim": 256,
                "dataset_root": str(DATASET_ROOT),
                "best_epoch": best_epoch,
            }

            torch.save(checkpoint, MODEL_PATH)

            print("\n✓ BEST MODEL SAVED")

            print(f"Path:\n{MODEL_PATH}")

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DINOv2 TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nBest validation loss:" f" {best_val_loss:.6f}")

    print(f"Best epoch:" f" {best_epoch}")

    print(f"\nModel saved at:" f"\n{MODEL_PATH}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
