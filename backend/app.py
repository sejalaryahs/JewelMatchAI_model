from pathlib import Path

from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
)
from flask_cors import CORS
from PIL import Image

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
            "message": ("Jewellery Matcher API is running"),
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
                        "error": ("No image was uploaded."),
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
                        "error": ("Source must be " "'gold' or 'prototype'."),
                    }
                ),
                400,
            )

        image = Image.open(image_file.stream).convert("RGB")

        # Current preprocessing layer.
        image = prepare_ring_image(image)

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
# SERVE CATALOGUE IMAGES
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
