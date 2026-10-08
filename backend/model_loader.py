from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "jewellery_dinov2.pth"
)

DINO_MODEL_NAME = "facebook/dinov2-base"

EMBEDDING_DIM = 512


class JewelleryDINOv2(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            DINO_MODEL_NAME
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(
                768,
                EMBEDDING_DIM,
            ),
            nn.LayerNorm(
                EMBEDDING_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                EMBEDDING_DIM,
                EMBEDDING_DIM,
            ),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM,
                EMBEDDING_DIM,
            ),
            nn.GELU(),
            nn.Linear(
                EMBEDDING_DIM,
                256,
            ),
        )

    def forward(
        self,
        pixel_values,
    ):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        cls_token = (
            outputs.last_hidden_state[:, 0, :]
        )

        embedding = self.embedding_head(
            cls_token
        )

        embedding = F.normalize(
            embedding,
            dim=1,
        )

        projection = self.projection_head(
            embedding
        )

        projection = F.normalize(
            projection,
            dim=1,
        )

        return embedding, projection


class ModelLoader:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print()
        print("=" * 70)
        print("LOADING JEWELLERY DINOv2")
        print("=" * 70)

        print(
            f"Device: {self.device}"
        )

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Trained model not found:\n"
                f"{MODEL_PATH}\n\n"
                f"Train the model first."
            )

        self.processor = (
            AutoImageProcessor.from_pretrained(
                DINO_MODEL_NAME
            )
        )

        self.model = JewelleryDINOv2()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Jewellery DINOv2 loaded successfully."
        )

        print("=" * 70)

    @torch.no_grad()
    def get_embedding(
        self,
        image,
    ):

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        embedding, _ = self.model(
            pixel_values
        )

        return embedding[0].cpu().numpy()