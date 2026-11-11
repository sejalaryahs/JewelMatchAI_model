import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.config import EMBEDDING_DIM
from backend.matching_service import JewelleryMatchingService

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_VERSION = "jewellery_model3_onnx_v1"

EMBEDDING_DIMENSION = EMBEDDING_DIM

TOP_K = 12

DEFAULT_QUERY_TYPE = "gold"

SUPPORTED_QUERY_TYPES = {
    "gold",
    "prototype",
}


# ============================================================
# MATCHING SERVICE
# ============================================================

matching_service = None


def initialize_matching_service():

    global matching_service

    print()
    print("=" * 70)
    print("JEWELLERY MODEL 3 API")
    print("=" * 70)

    matching_service = JewelleryMatchingService()

    print()
    print("Matching service initialized successfully.")
    print("Runtime: ONNX Runtime")
    print("Embedding dimension:", EMBEDDING_DIMENSION)
    print("Number of results:", TOP_K)
    print("Default query type:", DEFAULT_QUERY_TYPE)
    print("Matching mode: Bidirectional")

    print("=" * 70)


# ============================================================
# JSON SERIALIZATION
# ============================================================


def make_json_safe(value):

    if isinstance(value, dict):

        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):

        return [make_json_safe(item) for item in value]

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):

        return value

    return str(value)


# ============================================================
# HEALTH CHECK
# ============================================================


@app.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "success": True,
            "message": "Jewellery Model 3 API",
            "status": "running",
            "model": MODEL_VERSION,
            "runtime": "ONNX Runtime",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "top_k": TOP_K,
            "default_query_type": DEFAULT_QUERY_TYPE,
            "matching_mode": "Bidirectional",
            "supported_directions": [
                "Gold -> Prototype",
                "Prototype -> Gold",
            ],
            "matching_service": (
                "initialized" if matching_service is not None else "not initialized"
            ),
            "database": "Supabase",
        }
    )


# ============================================================
# MATCH JEWELLERY
# ============================================================


@app.route("/api/match", methods=["POST"])
def match_jewellery():

    try:

        # ----------------------------------------------------
        # CHECK MATCHING SERVICE
        # ----------------------------------------------------

        if matching_service is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Matching service is not initialized.",
                    }
                ),
                500,
            )

        # ----------------------------------------------------
        # CHECK IMAGE UPLOAD
        # ----------------------------------------------------

        if "image" not in request.files:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Please upload a jewellery image.",
                    }
                ),
                400,
            )

        uploaded_file = request.files["image"]

        if not uploaded_file.filename:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No image selected.",
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # GET QUERY TYPE
        #
        # Gold is the default for the current testing frontend.
        # ----------------------------------------------------

        query_type = (
            request.form.get(
                "query_type",
                DEFAULT_QUERY_TYPE,
            )
            .strip()
            .lower()
        )

        if not query_type:

            query_type = DEFAULT_QUERY_TYPE

        if query_type not in SUPPORTED_QUERY_TYPES:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": ("Invalid query_type. " "Use 'gold' or 'prototype'."),
                        "supported_types": [
                            "gold",
                            "prototype",
                        ],
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # DETERMINE TARGET TYPE
        # ----------------------------------------------------

        if query_type == "gold":

            target_type = "prototype"
            target_collection = "Prototype"

        else:

            target_type = "gold"
            target_collection = "Gold"

        # ----------------------------------------------------
        # VALIDATE IMAGE
        # ----------------------------------------------------

        try:

            image = Image.open(uploaded_file.stream)

            image.load()

            image = ImageOps.exif_transpose(image)

            image = image.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid image file.",
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # SAVE TEMPORARY QUERY IMAGE
        # ----------------------------------------------------

        with tempfile.TemporaryDirectory(prefix="jewellery_query_") as temp_dir:

            image_path = Path(temp_dir) / "query_image.jpg"

            image.save(
                image_path,
                format="JPEG",
                quality=95,
            )

            # ------------------------------------------------
            # MATCH USING ONNX MODEL AND SUPABASE EMBEDDINGS
            # ------------------------------------------------

            results = matching_service.match_image(
                image_path=image_path,
                query_type=query_type,
                top_k=TOP_K,
            )

        # ----------------------------------------------------
        # FORMAT RESULTS
        # ----------------------------------------------------

        formatted_results = []

        for result in results:

            result = make_json_safe(result)

            jewellery_id = result.get("id")

            name = result.get(
                "name",
                "Jewellery",
            )

            design_code = result.get(
                "design_code",
            )

            collection = result.get(
                "collection",
            )

            jewellery_type = result.get(
                "type",
            )

            description = result.get(
                "description",
            )

            # ------------------------------------------------
            # SUPABASE IMAGE
            # ------------------------------------------------

            image_url = result.get(
                "image_url",
            )

            image_path = result.get(
                "image_path",
            )

            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            similarity = float(
                result.get(
                    "similarity",
                    0.0,
                )
            )

            # ------------------------------------------------
            # RESULT OBJECT
            # ------------------------------------------------

            formatted_results.append(
                {
                    "id": str(jewellery_id if jewellery_id is not None else ""),
                    "name": name,
                    "design_code": design_code,
                    "collection": collection,
                    "type": jewellery_type,
                    "description": description,
                    "image": image_url,
                    "image_url": image_url,
                    "image_path": image_path,
                    "similarity": similarity,
                    "similarity_percentage": round(
                        similarity * 100,
                        2,
                    ),
                }
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response_data = {
            "success": True,
            "query_type": query_type,
            "target_type": target_type,
            "collection": target_collection,
            "embedding_model": MODEL_VERSION,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "runtime": "ONNX Runtime",
            "database": "Supabase",
            "requested_results": TOP_K,
            "count": len(formatted_results),
            "results": formatted_results,
        }

        return jsonify(make_json_safe(response_data))

    except Exception:

        app.logger.exception("Jewellery matching failed")

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Jewellery matching failed.",
                }
            ),
            500,
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("JEWELLERY MODEL 3 BACKEND")
    print("=" * 70)

    initialize_matching_service()

    app.run(
        host="0.0.0.0",
        port=5002,
        debug=False,
    )
