import tempfile
from pathlib import Path

import torch

from flask import Flask, jsonify, request
from flask_cors import CORS

from PIL import Image, UnidentifiedImageError

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

MODEL_VERSION = "jewellery_dinov2_v3"

EMBEDDING_DIMENSION = 512


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
            "embedding_dimension": EMBEDDING_DIMENSION,
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
        # VALIDATE IMAGE
        # ----------------------------------------------------

        try:

            image = Image.open(uploaded_file.stream)

            image.load()

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

        with tempfile.TemporaryDirectory() as temp_dir:

            image_path = Path(temp_dir) / "query_image.jpg"

            image.save(
                image_path,
                format="JPEG",
                quality=95,
            )

            # ------------------------------------------------
            # MATCH USING SUPABASE EMBEDDINGS
            # ------------------------------------------------

            results = matching_service.match_image(
                image_path=image_path,
                top_k=5,
            )

        # ----------------------------------------------------
        # FORMAT RESULTS
        # ----------------------------------------------------

        formatted_results = []

        for result in results:

            result = make_json_safe(result)

            # ------------------------------------------------
            # JEWELLERY INFORMATION
            # ------------------------------------------------

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

            image_url = result.get("image_url")

            image_path = result.get("image_path")

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
            # RESULT
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
            "query_type": "gold",
            "collection": "prototypes",
            "embedding_model": MODEL_VERSION,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "database": "Supabase",
            "count": len(formatted_results),
            "results": formatted_results,
        }

        return jsonify(make_json_safe(response_data))

    except Exception as error:

        app.logger.exception("Jewellery matching failed")

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(error),
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
