from pathlib import Path

import torch

from .config import MODEL_PATH, DEVICE

from training.model import JewelleryDINOv2

# ============================================================
# JEWELLERY MODEL LOADER
# ============================================================


class ModelLoader:

    def __init__(self):

        print()
        print("=" * 70)
        print("JEWELLERY MODEL LOADER")
        print("=" * 70)

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        if DEVICE == "cuda" and torch.cuda.is_available():

            self.device = torch.device("cuda")

        else:

            self.device = torch.device("cpu")

        print("Device:", self.device)

        # ----------------------------------------------------
        # MODEL PATH
        # ----------------------------------------------------

        self.model_path = Path(MODEL_PATH)

        print("Checkpoint:", self.model_path)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = None

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        self.checkpoint = None

        print("=" * 70)

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    def check_model_file(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model checkpoint not found: " f"{self.model_path}"
            )

        if not self.model_path.is_file():

            raise ValueError(f"Model path is not a file: " f"{self.model_path}")

        return True

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def load_checkpoint(self):

        print()

        print("Loading checkpoint...")

        self.check_model_file()

        try:

            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )

        except Exception as error:

            raise RuntimeError(f"Failed to load checkpoint: {error}") from error

        # ----------------------------------------------------
        # VALIDATE CHECKPOINT
        # ----------------------------------------------------

        if not isinstance(checkpoint, dict):

            raise ValueError("Invalid checkpoint format. " "Expected a dictionary.")

        if "model_state_dict" not in checkpoint:

            raise KeyError("Checkpoint does not contain " "'model_state_dict'.")

        self.checkpoint = checkpoint

        # ----------------------------------------------------
        # CHECKPOINT INFORMATION
        # ----------------------------------------------------

        print()

        print("Checkpoint loaded successfully.")

        print(
            "Epoch:",
            checkpoint.get(
                "epoch",
                "Unknown",
            ),
        )

        print(
            "Training loss:",
            checkpoint.get(
                "loss",
                "Unknown",
            ),
        )

        print(
            "Model name:",
            checkpoint.get(
                "model_name",
                "Unknown",
            ),
        )

        print(
            "Projection dimension:",
            checkpoint.get(
                "projection_dim",
                "Unknown",
            ),
        )

        return checkpoint

    # ========================================================
    # INITIALIZE MODEL
    # ========================================================

    def initialize_model(self):

        print()

        print("Initializing JewelleryDINOv2...")

        # ----------------------------------------------------
        # CREATE MODEL ARCHITECTURE
        # ----------------------------------------------------

        model = JewelleryDINOv2(
            freeze_backbone=True,
        )

        model = model.to(self.device)

        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------

        checkpoint = self.load_checkpoint()

        state_dict = checkpoint["model_state_dict"]

        # ----------------------------------------------------
        # LOAD TRAINED WEIGHTS
        # ----------------------------------------------------

        try:

            model.load_state_dict(
                state_dict,
                strict=True,
            )

        except RuntimeError as error:

            raise RuntimeError(
                "Checkpoint weights do not match "
                "the JewelleryDINOv2 architecture. "
                "Check the model configuration and "
                "checkpoint file.\n"
                f"Details: {error}"
            ) from error

        # ----------------------------------------------------
        # EVALUATION MODE
        # ----------------------------------------------------

        model.eval()

        self.model = model

        print()

        print("Model initialized successfully.")

        print("Model:", type(self.model).__name__)

        print("Device:", self.device)

        print("Model ready for inference.")

        print("=" * 70)

        return self.model

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):
        """
        Initialize the architecture and load trained weights.
        """

        return self.initialize_model()

    # ========================================================
    # GET MODEL
    # ========================================================

    def get_model(self):

        if self.model is None:

            raise RuntimeError("Model is not loaded. " "Call load_model() first.")

        return self.model

    # ========================================================
    # GET DEVICE
    # ========================================================

    def get_device(self):

        return self.device

    # ========================================================
    # GET CHECKPOINT INFORMATION
    # ========================================================

    def get_checkpoint_info(self):

        if self.checkpoint is None:

            return {}

        return {
            "epoch": self.checkpoint.get("epoch"),
            "loss": self.checkpoint.get("loss"),
            "model_name": self.checkpoint.get("model_name"),
            "projection_dim": self.checkpoint.get("projection_dim"),
        }


# ============================================================
# TEST MODEL LOADER
# ============================================================

if __name__ == "__main__":

    print()

    print("Testing Jewellery Model Loader...")

    loader = ModelLoader()

    model = loader.load_model()

    # --------------------------------------------------------
    # TEST IMAGE
    # --------------------------------------------------------

    dummy_image = torch.randn(
        1,
        3,
        224,
        224,
    ).to(loader.get_device())

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        embeddings = model(dummy_image)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("MODEL TEST RESULTS")

    print("=" * 70)

    print("Input shape:", tuple(dummy_image.shape))

    print("Embedding shape:", tuple(embeddings.shape))

    print("Embedding dimension:", embeddings.shape[1])

    print(
        "Embedding norm:",
        embeddings.norm(
            p=2,
            dim=1,
        ).tolist(),
    )

    print()

    print("Checkpoint information:")

    print(loader.get_checkpoint_info())

    print()

    print("Model loader test completed successfully.")

    print("=" * 70)
