from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2.pth"
ONNX_PATH = PROJECT_ROOT / "models" / "jewellery_dinov2.onnx"

DINO_MODEL_NAME = "facebook/dinov2-base"

EMBEDDING_DIM = 512


# ============================================================
# PYTORCH MODEL
# Must exactly match the trained model
# ============================================================


class JewelleryDINOv2(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(DINO_MODEL_NAME)

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

        outputs = self.backbone(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]

        embedding = self.embedding_head(cls_token)

        embedding = F.normalize(embedding, dim=1)

        return embedding


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================


def load_pytorch_model():

    print()
    print("Loading PyTorch model...")

    model = JewelleryDINOv2()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("✓ PyTorch model loaded")

    return model


# ============================================================
# CREATE TEST INPUT
# ============================================================


def create_test_input():

    print()
    print("Creating test input...")

    # Fixed random seed so that the same input is used
    # every time this script is executed.
    torch.manual_seed(42)

    image_tensor = torch.randn(1, 3, 224, 224)

    print("Input shape:", tuple(image_tensor.shape))

    return image_tensor


# ============================================================
# PYTORCH EMBEDDING
# ============================================================


def get_pytorch_embedding(model, image_tensor):

    print()
    print("Generating PyTorch embedding...")

    with torch.no_grad():

        embedding = model(image_tensor)

    embedding = embedding.numpy()

    print("PyTorch embedding shape:", embedding.shape)

    return embedding


# ============================================================
# ONNX SESSION
# ============================================================


def load_onnx_model():

    print()
    print("Loading ONNX model...")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    print("✓ ONNX model loaded")

    print()
    print("ONNX inputs:")

    for input_info in session.get_inputs():

        print(" ", input_info.name, input_info.shape, input_info.type)

    print()
    print("ONNX outputs:")

    for output_info in session.get_outputs():

        print(" ", output_info.name, output_info.shape, output_info.type)

    return session


# ============================================================
# ONNX EMBEDDING
# ============================================================


def get_onnx_embedding(session, image_tensor):

    print()
    print("Generating ONNX embedding...")

    input_array = image_tensor.numpy().astype(np.float32)

    outputs = session.run(["embedding"], {"pixel_values": input_array})

    embedding = outputs[0]

    print("ONNX embedding shape:", embedding.shape)

    return embedding


# ============================================================
# COMPARE
# ============================================================


def compare_embeddings(pytorch_embedding, onnx_embedding):

    print()
    print("=" * 70)
    print("EMBEDDING COMPARISON")
    print("=" * 70)

    pytorch_vector = pytorch_embedding[0]

    onnx_vector = onnx_embedding[0]

    # --------------------------------------------------------
    # Maximum absolute difference
    # --------------------------------------------------------

    max_difference = np.max(np.abs(pytorch_vector - onnx_vector))

    # --------------------------------------------------------
    # Mean absolute difference
    # --------------------------------------------------------

    mean_difference = np.mean(np.abs(pytorch_vector - onnx_vector))

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    pytorch_norm = pytorch_vector / np.linalg.norm(pytorch_vector)

    onnx_norm = onnx_vector / np.linalg.norm(onnx_vector)

    cosine_similarity = np.dot(pytorch_norm, onnx_norm)

    # --------------------------------------------------------
    # Euclidean distance
    # --------------------------------------------------------

    euclidean_distance = np.linalg.norm(pytorch_vector - onnx_vector)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(f"Maximum absolute difference : " f"{max_difference:.10f}")

    print(f"Mean absolute difference    : " f"{mean_difference:.10f}")

    print(f"Cosine similarity           : " f"{cosine_similarity:.10f}")

    print(f"Euclidean distance          : " f"{euclidean_distance:.10f}")

    print()

    # --------------------------------------------------------
    # Determine whether conversion is acceptable
    # --------------------------------------------------------

    if cosine_similarity >= 0.999:

        print("✓ EXCELLENT: ONNX and PyTorch " "embeddings are effectively equivalent.")

        result = True

    elif cosine_similarity >= 0.99:

        print("✓ GOOD: ONNX and PyTorch embeddings " "are very close.")

        result = True

    else:

        print("⚠ WARNING: ONNX and PyTorch embeddings " "differ more than expected.")

        result = False

    print()

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("JEWELLERY DINOv2 — PYTORCH vs ONNX")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"PyTorch model not found:\n{MODEL_PATH}")

    if not ONNX_PATH.exists():

        raise FileNotFoundError(f"ONNX model not found:\n{ONNX_PATH}")

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    pytorch_model = load_pytorch_model()

    onnx_session = load_onnx_model()

    # --------------------------------------------------------
    # Create identical input
    # --------------------------------------------------------

    test_input = create_test_input()

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    pytorch_embedding = get_pytorch_embedding(pytorch_model, test_input)

    onnx_embedding = get_onnx_embedding(onnx_session, test_input)

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    success = compare_embeddings(pytorch_embedding, onnx_embedding)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("=" * 70)

    if success:

        print("ONNX VALIDATION PASSED")

        print()
        print("The ONNX model can be used for " "the next deployment testing stage.")

    else:

        print("ONNX VALIDATION NEEDS INVESTIGATION")

    print("=" * 70)


if __name__ == "__main__":
    main()
