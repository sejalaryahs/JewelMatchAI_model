from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import MODEL_PATH, PROJECT_DIR, IMAGE_SIZE, EMBEDDING_DIM

# ============================================================
# JEWELLERY MODEL 3 - ONNX MODEL LOADER
# ============================================================


class ModelLoader:

    def __init__(self):

        print()
        print("=" * 70)
        print("JEWELLERY MODEL 3 - ONNX MODEL LOADER")
        print("=" * 70)

        # ----------------------------------------------------
        # MODEL PATH
        # ----------------------------------------------------

        self.model_path = Path(MODEL_PATH)

        # Resolve relative paths from the project directory.
        if not self.model_path.is_absolute():

            self.model_path = PROJECT_DIR / self.model_path

        self.model_path = self.model_path.resolve()

        print("ONNX Model:", self.model_path)

        # ----------------------------------------------------
        # ONNX SESSION
        # ----------------------------------------------------

        self.session = None

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        self.input_name = None
        self.output_name = None

        self.input_shape = None
        self.output_shape = None

        self.image_size = IMAGE_SIZE
        self.embedding_dim = EMBEDDING_DIM

        print("Image size:", self.image_size)
        print("Embedding dimension:", self.embedding_dim)

        print("=" * 70)

    # ========================================================
    # CHECK MODEL FILE
    # ========================================================

    def check_model_file(self):

        if not self.model_path.exists():

            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        if not self.model_path.is_file():

            raise ValueError(f"Model path is not a file: {self.model_path}")

        return True

    # ========================================================
    # LOAD ONNX MODEL
    # ========================================================

    def load_checkpoint(self):

        print()
        print("Loading ONNX model...")

        self.check_model_file()

        # ----------------------------------------------------
        # SESSION OPTIONS
        # ----------------------------------------------------

        session_options = ort.SessionOptions()

        session_options.intra_op_num_threads = 2

        session_options.inter_op_num_threads = 1

        # ----------------------------------------------------
        # LOAD ONNX SESSION
        # ----------------------------------------------------

        try:

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

        except Exception as error:

            raise RuntimeError(f"Failed to load ONNX model: {error}") from error

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        model_inputs = self.session.get_inputs()

        if not model_inputs:

            raise RuntimeError("ONNX model has no inputs.")

        self.input_name = model_inputs[0].name

        self.input_shape = model_inputs[0].shape

        # ----------------------------------------------------
        # MODEL OUTPUT
        # ----------------------------------------------------

        model_outputs = self.session.get_outputs()

        if not model_outputs:

            raise RuntimeError("ONNX model has no outputs.")

        self.output_name = model_outputs[0].name

        self.output_shape = model_outputs[0].shape

        # ----------------------------------------------------
        # VALIDATE MODEL DIMENSIONS
        # ----------------------------------------------------

        if len(self.input_shape) != 4:

            raise RuntimeError(
                "Expected ONNX input shape "
                "(batch, 3, height, width), "
                f"received {self.input_shape}"
            )

        if len(self.output_shape) != 2:

            raise RuntimeError(
                "Expected ONNX output shape "
                "(batch, embedding_dimension), "
                f"received {self.output_shape}"
            )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        print()
        print("ONNX model loaded successfully.")

        print("Input name:", self.input_name)

        print("Input shape:", self.input_shape)

        print("Output name:", self.output_name)

        print("Output shape:", self.output_shape)

        print("Execution providers:", self.session.get_providers())

        return self.session

    # ========================================================
    # INITIALIZE MODEL
    # ========================================================

    def initialize_model(self):

        print()
        print("Initializing ONNX model...")

        self.load_checkpoint()

        print()
        print("Model initialized successfully.")

        print("Model: Jewellery Model 3")

        print("Runtime: ONNX Runtime")

        print("Model ready for inference.")

        print("=" * 70)

        return self.session

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):
        """
        Load the Jewellery Model 3 ONNX model.
        """

        return self.initialize_model()

    # ========================================================
    # GET MODEL
    # ========================================================

    def get_model(self):

        if self.session is None:

            raise RuntimeError("Model is not loaded. Call load_model() first.")

        return self.session

    # ========================================================
    # GET EMBEDDINGS
    # ========================================================

    def get_embeddings(self, pixel_values):
        """
        Generate jewellery embeddings using ONNX Runtime.

        Parameters
        ----------
        pixel_values:
        NumPy array with shape:

            (batch, 3, IMAGE_SIZE, IMAGE_SIZE)

        Input must already be preprocessed and normalized
        according to the training preprocessing.

        Returns
        -------
        NumPy array with shape:

            (batch, EMBEDDING_DIM)
        """

        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        if self.session is None:

            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # ----------------------------------------------------
        # INPUT TYPE
        # ----------------------------------------------------

        if not isinstance(pixel_values, np.ndarray):

            raise TypeError("pixel_values must be a NumPy array.")

        # ----------------------------------------------------
        # INPUT DIMENSIONS
        # ----------------------------------------------------

        if pixel_values.ndim != 4:

            raise ValueError(
                "Expected input shape "
                "(batch, 3, IMAGE_SIZE, IMAGE_SIZE), "
                f"received {pixel_values.shape}"
            )

        expected_shape = (
            3,
            self.image_size,
            self.image_size,
        )

        if pixel_values.shape[1:] != expected_shape:

            raise ValueError(
                f"Expected image dimensions {expected_shape}, "
                f"received {pixel_values.shape[1:]}"
            )

        # ----------------------------------------------------
        # EMPTY BATCH
        # ----------------------------------------------------

        if pixel_values.shape[0] == 0:

            raise ValueError("Input batch cannot be empty.")

        # ----------------------------------------------------
        # FLOAT32 INPUT
        # ----------------------------------------------------

        pixel_values = np.asarray(
            pixel_values,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # VALIDATE INPUT VALUES
        # ----------------------------------------------------

        if not np.isfinite(pixel_values).all():

            raise ValueError("Input contains NaN or infinite values.")

        # ----------------------------------------------------
        # ONNX INFERENCE
        # ----------------------------------------------------

        try:

            embeddings = self.session.run(
                [self.output_name],
                {self.input_name: pixel_values},
            )[0]

        except Exception as error:

            raise RuntimeError(f"ONNX inference failed: {error}") from error

        # ----------------------------------------------------
        # OUTPUT TYPE
        # ----------------------------------------------------

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # OUTPUT DIMENSIONS
        # ----------------------------------------------------

        if embeddings.ndim != 2:

            raise RuntimeError("Unexpected embedding shape: " f"{embeddings.shape}")

        # ----------------------------------------------------
        # EMBEDDING DIMENSION
        # ----------------------------------------------------

        if embeddings.shape[1] != self.embedding_dim:

            raise RuntimeError(
                f"Expected {self.embedding_dim}-dimensional "
                f"embeddings, received {embeddings.shape[1]}"
            )

        # ----------------------------------------------------
        # OUTPUT BATCH VALIDATION
        # ----------------------------------------------------

        if embeddings.shape[0] != pixel_values.shape[0]:

            raise RuntimeError("Input and output batch sizes do not match.")

        # ----------------------------------------------------
        # OUTPUT VALIDATION
        # ----------------------------------------------------

        if not np.isfinite(embeddings).all():

            raise RuntimeError("Model output contains NaN or infinite values.")

        # ----------------------------------------------------
        # RETURN EMBEDDINGS
        # ----------------------------------------------------

        return embeddings

    # ========================================================
    # GET MODEL INFORMATION
    # ========================================================

    def get_model_info(self):

        return {
            "model": "Jewellery Model 3",
            "runtime": "ONNX Runtime",
            "model_path": str(self.model_path),
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "embedding_dimension": self.embedding_dim,
            "image_size": self.image_size,
            "providers": (
                self.session.get_providers() if self.session is not None else []
            ),
        }


# ============================================================
# TEST MODEL LOADER
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing Jewellery Model 3 ONNX Loader...")

    loader = ModelLoader()

    loader.load_model()

    # --------------------------------------------------------
    # TEST INPUT
    # --------------------------------------------------------

    dummy_image = np.random.randn(
        1,
        3,
        loader.image_size,
        loader.image_size,
    ).astype(np.float32)

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    embeddings = loader.get_embeddings(dummy_image)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("MODEL TEST RESULTS")

    print("=" * 70)

    print("Input shape:", dummy_image.shape)

    print("Embedding shape:", embeddings.shape)

    print("Embedding dimension:", embeddings.shape[1])

    print(
        "Embedding norm:",
        np.linalg.norm(
            embeddings,
            axis=1,
        ).tolist(),
    )

    print()

    print("Model information:")

    print(loader.get_model_info())

    print()

    print("ONNX model loader test completed successfully.")

    print("=" * 70)
