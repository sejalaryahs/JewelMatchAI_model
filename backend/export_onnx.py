from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2.onnx"

DINO_MODEL_NAME = "facebook/dinov2-base"

EMBEDDING_DIM = 512


# ============================================================
# MODEL
# Must exactly match the architecture used during training
# ============================================================

class JewelleryDINOv2(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            DINO_MODEL_NAME
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(768, EMBEDDING_DIM),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.GELU(),
            nn.Linear(EMBEDDING_DIM, 256),
        )

    def forward(self, pixel_values):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        # DINOv2 CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]

        # 512D embedding
        embedding = self.embedding_head(
            cls_token
        )

        # Normalize embedding
        embedding = F.normalize(
            embedding,
            dim=1
        )

        return embedding


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("JEWELLERY DINOv2 → ONNX EXPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check trained model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    print()
    print(f"Trained model:")
    print(MODEL_PATH)

    print()
    print("Output ONNX model:")
    print(ONNX_PATH)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cpu")

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading DINOv2...")

    model = JewelleryDINOv2()

    print("Loading trained checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("✓ Trained model loaded")

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    print()
    print("Creating dummy input...")

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    # --------------------------------------------------------
    # Test PyTorch output
    # --------------------------------------------------------

    print()
    print("Testing PyTorch model...")

    with torch.no_grad():

        pytorch_embedding = model(
            dummy_input
        )

    print(
        "PyTorch embedding shape:",
        tuple(pytorch_embedding.shape)
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    print()
    print("Exporting ONNX model...")

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),

        input_names=[
            "pixel_values"
        ],

        output_names=[
            "embedding"
        ],

        dynamic_axes={
            "pixel_values": {
                0: "batch_size"
            },
            "embedding": {
                0: "batch_size"
            },
        },

        opset_version=17,

        do_constant_folding=True,
    )

    print("✓ ONNX export completed")

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not ONNX_PATH.exists():
        raise RuntimeError(
            "ONNX file was not created."
        )

    size_mb = (
        ONNX_PATH.stat().st_size
        / (1024 * 1024)
    )

    print()
    print(
        f"ONNX model size: {size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # ONNX validation
    # --------------------------------------------------------

    print()
    print("Validating ONNX model...")

    import onnx

    onnx_model = onnx.load(
        str(ONNX_PATH)
    )

    onnx.checker.check_model(
        onnx_model
    )

    print("✓ ONNX model is valid")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPORT SUCCESSFUL")
    print("=" * 70)

    print()
    print("Created:")

    print(
        ONNX_PATH
    )

    print()
    print(
        "PyTorch embedding shape:",
        tuple(pytorch_embedding.shape)
    )

    print()
    print("Next step:")
    print(
        "Compare PyTorch and ONNX embeddings before deployment."
    )


if __name__ == "__main__":
    main()