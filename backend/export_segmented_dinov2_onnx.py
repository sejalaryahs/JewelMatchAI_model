import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
MODEL_DIR = ROOT / "models"

CHECKPOINT_PATH = MODEL_DIR / "jewellery_dinov2_segmented_experiment.pth"

# A new filename: do not overwrite the existing production ONNX.
ONNX_PATH = MODEL_DIR / "jewellery_dinov2_segmented_experiment.onnx"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model import JewelleryDINOv2

# ------------------------------------------------------------
# EXPORT ONLY THE EMBEDDING USED BY THE MATCHER
# ------------------------------------------------------------


class EmbeddingOnly(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model

    def forward(self, pixel_values):
        embedding, _projection = self.model(pixel_values)
        return embedding


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("=" * 65)
    print("EXPORT EXPERIMENTAL SEGMENTED DINOV2 TO ONNX")
    print("=" * 65)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Output:", ONNX_PATH)

    # Load only the trusted checkpoint from this project.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    print("Checkpoint model:", checkpoint.get("model_name"))
    print("Embedding dimension:", checkpoint.get("embedding_dim"))
    print("Projection dimension:", checkpoint.get("projection_dim"))

    if int(checkpoint.get("embedding_dim", -1)) != 512:
        raise ValueError(
            "Expected a 512-dimensional embedding. "
            "Check the checkpoint before exporting."
        )

    # Rebuild the exact architecture from training/model.py.
    trained_model = JewelleryDINOv2()
    trained_model.load_state_dict(checkpoint["model_state_dict"])
    trained_model.eval()

    export_model = EmbeddingOnly(trained_model).eval()

    # Model input is a preprocessed float32 image tensor:
    # [batch, 3, 224, 224].
    dummy_input = torch.randn(1, 3, 224, 224)

    # Confirm the expected output before export.
    with torch.inference_mode():
        pytorch_output = export_model(dummy_input)

    if tuple(pytorch_output.shape) != (1, 512):
        raise RuntimeError(f"Unexpected PyTorch output shape: {pytorch_output.shape}")

    # Export to a separate ONNX file.
    torch.onnx.export(
        export_model,
        (dummy_input,),
        str(ONNX_PATH),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
        external_data=True,
        do_constant_folding=True,
    )

    # Validate ONNX structure.
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model structure validated.")

    # Run CPU inference using ONNX Runtime.
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["embedding"],
        {"pixel_values": dummy_input.numpy()},
    )[0]

    pytorch_array = pytorch_output.detach().cpu().numpy()
    onnx_array = np.asarray(onnx_output)

    if pytorch_array.shape != onnx_array.shape:
        raise RuntimeError(
            f"Output shape mismatch: PyTorch {pytorch_array.shape}, "
            f"ONNX {onnx_array.shape}"
        )

    if not np.isfinite(onnx_array).all():
        raise RuntimeError("ONNX output contains NaN or infinity.")

    max_abs_diff = float(np.max(np.abs(pytorch_array - onnx_array)))
    mean_abs_diff = float(np.mean(np.abs(pytorch_array - onnx_array)))

    print()
    print("=" * 65)
    print("PYTORCH VS ONNX VALIDATION")
    print("=" * 65)
    print("PyTorch output shape:", pytorch_array.shape)
    print("ONNX output shape:", onnx_array.shape)
    print(f"Maximum absolute difference: {max_abs_diff:.8f}")
    print(f"Mean absolute difference:    {mean_abs_diff:.8f}")

    # Numerical tolerance may vary by hardware and ONNX operators.
    try:
        np.testing.assert_allclose(
            pytorch_array,
            onnx_array,
            rtol=1e-3,
            atol=1e-4,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "ONNX output differs beyond the selected tolerance. "
            "Do not switch the matcher yet; inspect the differences."
        ) from exc

    print("[SUCCESS] PyTorch and ONNX outputs agree within tolerance.")
    print("ONNX model:", ONNX_PATH)
    print("ONNX file size (MB):", round(ONNX_PATH.stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
