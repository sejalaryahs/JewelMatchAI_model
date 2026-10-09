from pathlib import Path

from bson import ObjectId
from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    send_from_directory,
)
from flask_cors import CORS
from gridfs import GridFS
from PIL import Image

from .database.mongodb import get_jewellery_collection
from .model_loader import ModelLoader
from .matcher import JewelleryMatcher
from .segmentation import prepare_ring_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATCHING_DATA_DIR = PROJECT_ROOT / "matching_data"


app = Flask(__name__)

CORS(app)


# ============================================================
# LOAD AI MODEL ONCE
# ============================================================

model_loader = ModelLoader()

matcher = JewelleryMatcher(model_loader)


# ============================================================
# HEALTH CHECK
# ============================================================


@app.route(
    "/api/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "status": "ok",
            "message": "Jewellery Matcher API is running",
        }
    )


# ============================================================
# MATCH
# ============================================================


@app.route(
    "/api/match",
    methods=["POST"],
)
def match():

    try:

        if "image" not in request.files:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No image was uploaded.",
                    }
                ),
                400,
            )

        image_file = request.files["image"]

        source = (
            request.form.get(
                "source",
                "prototype",
            )
            .strip()
            .lower()
        )

        top_k = int(
            request.form.get(
                "top_k",
                5,
            )
        )

        if source not in {
            "gold",
            "prototype",
        }:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Source must be 'gold' or 'prototype'.",
                    }
                ),
                400,
            )

        image = Image.open(image_file.stream).convert("RGB")

        # ----------------------------------------------------
        # CURRENT JEWELLERY IMAGE PREPROCESSING
        # ----------------------------------------------------

        image = prepare_ring_image(image)

        # ----------------------------------------------------
        # FIND MATCHES
        # ----------------------------------------------------

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
                "results": results,
            }
        )

    except Exception as exc:

        print(f"[ERROR] /api/match: {exc}")

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )


# ============================================================
# SERVE MONGODB / GRIDFS CATALOGUE IMAGE
# ============================================================


@app.route(
    "/catalogue-image/stored/<gridfs_id>",
    methods=["GET"],
)
def catalogue_image_stored(
    gridfs_id,
):

    try:

        # ----------------------------------------------------
        # VALIDATE MONGODB OBJECT ID
        # ----------------------------------------------------

        try:

            object_id = ObjectId(gridfs_id)

        except Exception:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid GridFS image ID.",
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # GET SAME MONGODB DATABASE
        # ----------------------------------------------------

        database = get_jewellery_collection().database

        # ----------------------------------------------------
        # GRIDFS
        # ----------------------------------------------------

        gridfs = GridFS(database)

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # READ GRIDFS FILE
        # ----------------------------------------------------

        grid_file = gridfs.get(object_id)

        # ----------------------------------------------------
        # DETERMINE CONTENT TYPE
        # ----------------------------------------------------

        content_type = getattr(
            grid_file,
            "content_type",
            None,
        )

        # ----------------------------------------------------
        # SEND IMAGE TO BROWSER
        # ----------------------------------------------------

        response = send_file(
            grid_file,
            mimetype=content_type,
            download_name=(
                grid_file.filename
                if getattr(
                    grid_file,
                    "filename",
                    None,
                )
                else "jewellery-image"
            ),
        )

        # ----------------------------------------------------
        # CACHE IMAGE
        # ----------------------------------------------------

        response.headers["Cache-Control"] = "public, max-age=31536000"

        return response

    except Exception as exc:

        print(
            "[ERROR] GridFS image error:",
            repr(exc),
        )

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )


# ============================================================
# OLD LOCAL CATALOGUE IMAGE ROUTE
#
# Kept temporarily for backwards compatibility.
# The React application now uses the MongoDB/GridFS route
# above.
# ============================================================


@app.route(
    "/catalogue/<domain>/<filename>",
    methods=["GET"],
)
def catalogue_image(
    domain,
    filename,
):

    if domain not in {
        "gold",
        "prototypes",
    }:

        return (
            jsonify({"error": "Invalid domain."}),
            400,
        )

    directory = MATCHING_DATA_DIR / domain

    return send_from_directory(
        directory,
        filename,
    )


# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
