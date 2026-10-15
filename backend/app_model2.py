from io import BytesIO

from bson import ObjectId
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from gridfs import GridFS
from PIL import Image, UnidentifiedImageError

from .database.mongodb import (
    get_database,
    test_mongodb_connection,
)
from .onnx_matcher import ONNXJewelleryMatcher, MODEL_ID

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# This service loads only the Model 2 ONNX model.
matcher = ONNXJewelleryMatcher()


@app.errorhandler(413)
def request_too_large(_error):
    return (
        jsonify(
            {
                "status": "error",
                "message": "Image is too large. Maximum upload size is 10 MB.",
            }
        ),
        413,
    )


@app.get("/")
def index():
    return jsonify(
        {
            "service": "JewelMatchAI Model 2",
            "model": "Segmented DINOv2 ONNX",
            "model_id": MODEL_ID,
            "status": "running",
        }
    )


@app.get("/api/health")
def health():
    try:
        database_ok = bool(test_mongodb_connection())
    except Exception as exc:
        app.logger.exception("MongoDB health check failed")
        return (
            jsonify(
                {
                    "status": "error",
                    "model": "segmented_dinov2_onnx",
                    "model_id": MODEL_ID,
                    "mongodb": "unavailable",
                    "message": str(exc),
                }
            ),
            503,
        )

    if not database_ok:
        return (
            jsonify(
                {
                    "status": "error",
                    "model": "segmented_dinov2_onnx",
                    "model_id": MODEL_ID,
                    "mongodb": "unavailable",
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "model": "segmented_dinov2_onnx",
            "model_id": MODEL_ID,
            "mongodb": "connected",
            "provider": "CPUExecutionProvider",
        }
    )


@app.route(
    "/api/match/experimental",
    methods=["POST"],
)
@app.route("/api/match", methods=["POST"])
def match_jewellery():
    """
    Multipart/form-data:
      image: uploaded jewellery image
      source: 'gold' or 'prototype'
      top_k: optional number of matches
    """
    if "image" not in request.files:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Upload an image using the 'image' field.",
                }
            ),
            400,
        )

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "No image file was selected.",
                }
            ),
            400,
        )

    source = request.form.get("source", "prototype").strip().lower()
    top_k = request.form.get("top_k", 5)

    if source not in ("gold", "prototype"):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "source must be 'gold' or 'prototype'.",
                }
            ),
            400,
        )

    try:
        with Image.open(uploaded_file.stream) as opened_image:
            image = opened_image.convert("RGB")

        matches = matcher.find_matches(
            image=image,
            source=source,
            top_k=top_k,
        )

        return jsonify(
            {
                "status": "success",
                "model_type": "segmented_dinov2_onnx",
                "model_id": MODEL_ID,
                "source": source,
                "target": "prototype" if source == "gold" else "gold",
                "count": len(matches),
                "matches": matches,
            }
        )

    except UnidentifiedImageError:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "The uploaded file is not a valid image.",
                }
            ),
            400,
        )

    except ValueError as exc:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                }
            ),
            400,
        )

    except Exception as exc:
        app.logger.exception("Model 2 matching failed")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Matching failed. Check the service logs.",
                }
            ),
            500,
        )


@app.get("/catalogue-image/stored/<gridfs_id>")
def stored_catalogue_image(gridfs_id):
    """Serve an image stored in MongoDB GridFS."""
    try:
        object_id = ObjectId(gridfs_id)
        database = get_database()
        gridfs = GridFS(database)

        if not gridfs.exists(object_id=object_id):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Catalogue image not found.",
                    }
                ),
                404,
            )

        stored_file = gridfs.get(object_id)
        image_bytes = stored_file.read()

        return send_file(
            BytesIO(image_bytes),
            mimetype=stored_file.content_type or "image/jpeg",
            download_name=stored_file.filename or "catalogue-image.jpg",
            max_age=3600,
        )

    except Exception:
        app.logger.exception("Could not retrieve GridFS image")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Could not retrieve the catalogue image.",
                }
            ),
            500,
        )


@app.post("/api/model2/cache/clear")
def clear_model2_cache():
    """
    Clear cached MongoDB embeddings after regenerating Model 2 vectors.

    Protect this endpoint before exposing it publicly if you use it.
    """
    matcher.clear_catalogue_cache()

    return jsonify(
        {
            "status": "success",
            "message": "Model 2 catalogue embedding cache cleared.",
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
