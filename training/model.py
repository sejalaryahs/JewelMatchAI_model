import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from config import (
    DINO_MODEL_NAME,
    EMBEDDING_DIM,
    PROJECTION_DIM,
)

# ============================================================
# DINOv2 JEWELLERY MODEL
# ============================================================


class JewelleryDINOv2(nn.Module):

    def __init__(
        self,
        freeze_backbone=True,
    ):

        super().__init__()

        # ----------------------------------------------------
        # LOAD PRETRAINED DINOv2
        # ----------------------------------------------------

        print()

        print("=" * 70)

        print("LOADING PRETRAINED DINOv2")

        print("=" * 70)

        print(f"Model: {DINO_MODEL_NAME}")

        self.backbone = AutoModel.from_pretrained(DINO_MODEL_NAME)

        # ----------------------------------------------------
        # DINOv2 HIDDEN SIZE
        # ----------------------------------------------------

        backbone_dim = self.backbone.config.hidden_size

        if backbone_dim != 768:

            raise RuntimeError(
                "Unexpected DINOv2 hidden size. " f"Expected 768, got {backbone_dim}."
            )

        # ----------------------------------------------------
        # EMBEDDING HEAD
        # ----------------------------------------------------

        self.embedding_head = nn.Sequential(
            nn.Linear(
                backbone_dim,
                EMBEDDING_DIM,
            ),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(
                EMBEDDING_DIM,
                EMBEDDING_DIM,
            ),
        )

        # ----------------------------------------------------
        # PROJECTION HEAD
        # ----------------------------------------------------

        self.projection_head = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM,
                EMBEDDING_DIM,
            ),
            nn.GELU(),
            nn.Linear(
                EMBEDDING_DIM,
                PROJECTION_DIM,
            ),
        )

        # ----------------------------------------------------
        # FREEZE BACKBONE
        # ----------------------------------------------------

        if freeze_backbone:

            self.freeze_backbone()

    # ========================================================
    # FREEZE DINOv2
    # ========================================================

    def freeze_backbone(self):

        for parameter in self.backbone.parameters():

            parameter.requires_grad = False

    # ========================================================
    # UNFREEZE FINAL LAYERS
    # ========================================================

    def unfreeze_last_layers(
        self,
        number_of_layers=4,
    ):

        self.freeze_backbone()

        encoder_layers = self.backbone.encoder.layer

        total_layers = len(encoder_layers)

        start_index = max(
            0,
            total_layers - number_of_layers,
        )

        print()

        print(f"Unfreezing final " f"{number_of_layers} DINOv2 layers...")

        for layer in encoder_layers[start_index:]:

            for parameter in layer.parameters():

                parameter.requires_grad = True

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        pixel_values,
    ):

        outputs = self.backbone(pixel_values=pixel_values)

        # ----------------------------------------------------
        # CLS TOKEN
        # ----------------------------------------------------

        cls_token = outputs.last_hidden_state[:, 0, :]

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding = self.embedding_head(cls_token)

        embedding = F.normalize(
            embedding,
            dim=1,
        )

        # ----------------------------------------------------
        # PROJECTION
        # ----------------------------------------------------

        projection = self.projection_head(embedding)

        projection = F.normalize(
            projection,
            dim=1,
        )

        return (
            embedding,
            projection,
        )

    # ========================================================
    # EMBEDDING ONLY
    # ========================================================

    @torch.no_grad()
    def get_embedding(
        self,
        pixel_values,
    ):

        outputs = self.backbone(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]

        embedding = self.embedding_head(cls_token)

        embedding = F.normalize(
            embedding,
            dim=1,
        )

        return embedding
