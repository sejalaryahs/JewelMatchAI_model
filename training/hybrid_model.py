import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

from model import JewelleryDINOv2
from config import EMBEDDING_DIM, PROJECTION_DIM


class JewelleryHybridModel(nn.Module):
    """
    Combines DINOv2 and ResNet18 features.

    DINOv2:
        Learns rich visual representations.

    ResNet18:
        Adds complementary convolutional features.

    Fusion:
        Produces a normalized 512-dimensional embedding.

    The original JewelleryDINOv2 class remains unchanged.
    """

    def __init__(
        self,
        freeze_backbone=True,
        pretrained_resnet=True,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. Existing DINOv2 branch
        # --------------------------------------------------

        self.dino_model = JewelleryDINOv2(freeze_backbone=freeze_backbone)

        # --------------------------------------------------
        # 2. ResNet18 branch
        # --------------------------------------------------

        weights = ResNet18_Weights.DEFAULT if pretrained_resnet else None

        self.resnet = resnet18(weights=weights)

        resnet_feature_dim = self.resnet.fc.in_features

        # Remove the ImageNet classification layer.
        self.resnet.fc = nn.Identity()

        self.resnet_projection = nn.Sequential(
            nn.Linear(resnet_feature_dim, EMBEDDING_DIM),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.GELU(),
        )

        # --------------------------------------------------
        # 3. Feature fusion
        # --------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

        # --------------------------------------------------
        # 4. Contrastive projection head
        # --------------------------------------------------

        self.projection_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.GELU(),
            nn.Linear(EMBEDDING_DIM, PROJECTION_DIM),
        )

    def freeze_backbone(self):
        """Freeze DINOv2 backbone parameters."""

        self.dino_model.freeze_backbone()

    def unfreeze_last_layers(self, number_of_layers=4):
        """Unfreeze the last DINOv2 transformer layers."""

        self.dino_model.unfreeze_last_layers(number_of_layers)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values:
                Tensor shaped [batch, 3, 224, 224].

        Returns:
            embedding:
                Normalized [batch, 512] fused embeddings.

            projection:
                Normalized [batch, 256] contrastive projections.
        """

        # DINOv2 features
        dino_embedding, _ = self.dino_model(pixel_values)

        # ResNet18 features
        resnet_features = self.resnet(pixel_values)

        resnet_embedding = self.resnet_projection(resnet_features)

        # Combine both feature representations
        combined_features = torch.cat(
            [dino_embedding, resnet_embedding],
            dim=1,
        )

        # Final retrieval embedding
        embedding = self.fusion(combined_features)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        # Training-only contrastive projection
        projection = self.projection_head(embedding)

        projection = F.normalize(
            projection,
            p=2,
            dim=1,
        )

        return embedding, projection

    @torch.no_grad()
    def get_embedding(self, pixel_values):
        """Return the fused embedding for matching."""

        embedding, _ = self.forward(pixel_values)

        return embedding
