import gc
import os
from pathlib import Path

from bson import ObjectId
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from gridfs import GridFS
from PIL import Image

from .database.mongodb import get_database, test_mongodb_connection
from .onnx_matcher import ONNXJewelleryMatcher

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

matcher = ONNXJewelleryMatcher()


@app.route("/api/health", methods=["GET"])
def health():
    try:
        test_mongodb_connection()
        return jsonify(
            {
                "status": "ok",
                "model": "segmented_dinov2_onnx",
                "model_id": matcher.__class__.__name__,
            }
        )
    except Exception:
        app.logger.exception("Health check failed.")
        return jsonify({"status": "error"}), 503


@app.route("/api/match/experimental", methods=["POST"])
@app.route("/api/match", methods=["POST"])
def match():
    image = None
    uploaded = None

    try:
        uploaded = request.files.get("image")

        if uploaded is None or not uploaded.filename:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Please upload an image.",
                    }
                ),
                400,
            )

        source = request.form.get("source", "prototype").lower().strip()

        if source not in {"gold", "prototype"}:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Source must be 'gold' or 'prototype'.",
                    }
                ),
                400,
            )

        try:
            top_k = int(request.form.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5

        with Image.open(uploaded.stream) as original:
            image = original.convert("RGB")

        results = matcher.find_matches(
            image=image,
            source=source,
            top_k=top_k,
        )

        target = "prototype" if source == "gold" else "gold"

        return jsonify(
            {
                "success": True,
                "source": source,
                "target": target,
                "model_type": "segmented_dinov2_onnx",
                "results": results,
            }
        )

    except ValueError as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            400,
        )

    except Exception:
        app.logger.exception("Model 2 matching failed.")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Model 2 matching failed.",
                }
            ),
            500,
        )

    finally:
        if image is not None:
            image.close()

        if uploaded is not None:
            uploaded.close()

        gc.collect()


@app.route("/catalogue-image/stored/<gridfs_id>", methods=["GET"])
def catalogue_image_stored(gridfs_id):
    try:
        object_id = ObjectId(gridfs_id)
        gridfs = GridFS(get_database())

        if not gridfs.exists(object_id):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Stored image not found.",
                    }
                ),
                404,
            )

        grid_file = gridfs.get(object_id)

        return send_file(
            grid_file,
            mimetype=(
                getattr(grid_file, "content_type", None) or "application/octet-stream"
            ),
            download_name=(getattr(grid_file, "filename", None) or "jewellery-image"),
        )

    except Exception:
        app.logger.exception("GridFS retrieval failed.")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Could not retrieve the stored image.",
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
