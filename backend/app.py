import gc
import threading
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
from .experimental_matcher import ExperimentalJewelleryMatcher
from .fast_matcher import FastJewelleryMatcher

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCHING_DATA_DIR = PROJECT_ROOT / "matching_data"


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)

# Maximum uploaded image size: 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# LOAD EXISTING MATCHERS
#
# Existing production and experimental implementations are
# preserved. MobileNet is loaded lazily by get_fast_matcher().
# ============================================================

model_loader = ModelLoader()
matcher = JewelleryMatcher(model_loader)

experimental_matcher = ExperimentalJewelleryMatcher()


# ============================================================
# LAZY-LOAD FAST MATCHER
# ============================================================

_fast_matcher = None
_fast_matcher_lock = threading.Lock()


def get_fast_matcher():
    """Initialize MobileNet only when the fast endpoint is used."""
    global _fast_matcher

    if _fast_matcher is None:
        with _fast_matcher_lock:
            if _fast_matcher is None:
                _fast_matcher = FastJewelleryMatcher(device="cpu")

    return _fast_matcher


# ============================================================
# HEALTH CHECK
# ============================================================


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Jewellery Matcher API is running",
        }
    )


# ============================================================
# PRODUCTION MATCHER
# POST /api/match
# ============================================================


@app.route("/api/match", methods=["POST"])
def match():
    image = None
    image_file = None

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

        if not image_file.filename:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No image was selected.",
                    }
                ),
                400,
            )

        source = request.form.get("source", "prototype").strip().lower()

        try:
            top_k = int(request.form.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5

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

        with Image.open(image_file.stream) as uploaded_image:
            image = uploaded_image.convert("RGB")

        # Preserve existing production preprocessing.
        processed_image = prepare_ring_image(image)

        try:
            results = matcher.find_matches(
                image=processed_image,
                source=source,
                top_k=top_k,
            )
        finally:
            if processed_image is not image:
                try:
                    processed_image.close()
                except Exception:
                    pass

        target = "prototype" if source == "gold" else "gold"

        return jsonify(
            {
                "success": True,
                "source": source,
                "target": target,
                "results": results,
            }
        )

    except Exception:
        app.logger.exception("Production matching failed.")

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Matching failed. Please try again.",
                }
            ),
            500,
        )

    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass

        if image_file is not None:
            try:
                image_file.close()
            except Exception:
                pass

        gc.collect()


# ============================================================
# DINOv2 EXPERIMENTAL MATCHER
# POST /api/match/experimental
# ============================================================


@app.route("/api/match/experimental", methods=["POST"])
def match_experimental():
    image = None
    image_file = None

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

        if not image_file.filename:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No image was selected.",
                    }
                ),
                400,
            )

        source = request.form.get("source", "prototype").strip().lower()

        try:
            top_k = int(request.form.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5

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

        with Image.open(image_file.stream) as uploaded_image:
            image = uploaded_image.convert("RGB")

        results = experimental_matcher.find_matches(
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
                "model_type": "experimental_segmented_dinov2",
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
        app.logger.exception("DINOv2 experimental matching failed.")

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Experimental matching failed.",
                }
            ),
            500,
        )

    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass

        if image_file is not None:
            try:
                image_file.close()
            except Exception:
                pass

        gc.collect()


# ============================================================
# FAST MOBILENET MATCHER
# POST /api/match/fast
# ============================================================


@app.route("/api/match/fast", methods=["POST"])
def match_fast():
    image = None
    image_file = None

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

        if not image_file.filename:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No image was selected.",
                    }
                ),
                400,
            )

        source = request.form.get("source", "prototype").strip().lower()

        try:
            top_k = int(request.form.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5

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

        with Image.open(image_file.stream) as uploaded_image:
            image = uploaded_image.convert("RGB")

        # Do not apply the DINOv2 segmentation pipeline here.
        # The fast matcher uses its own consistent preprocessing.
        fast_matcher = get_fast_matcher()

        results = fast_matcher.find_matches(
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
                "model_type": "mobilenet_v3_small",
                "count": len(results),
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
        app.logger.exception("Fast MobileNet matching failed.")

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Fast matching failed. Please try again.",
                }
            ),
            500,
        )

    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass

        if image_file is not None:
            try:
                image_file.close()
            except Exception:
                pass

        gc.collect()


# ============================================================
# SERVE MONGODB / GRIDFS CATALOGUE IMAGE
# GET /catalogue-image/stored/<gridfs_id>
# ============================================================


@app.route(
    "/catalogue-image/stored/<gridfs_id>",
    methods=["GET"],
)
def catalogue_image_stored(gridfs_id):
    try:
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

        database = get_jewellery_collection().database
        gridfs = GridFS(database)

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

        response = send_file(
            grid_file,
            mimetype=(
                getattr(grid_file, "content_type", None) or "application/octet-stream"
            ),
            download_name=(
                grid_file.filename
                if getattr(grid_file, "filename", None)
                else "jewellery-image"
            ),
        )

        response.headers["Cache-Control"] = "public, max-age=31536000"

        return response

    except Exception:
        app.logger.exception("GridFS image retrieval failed.")

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Could not retrieve the stored image.",
                }
            ),
            500,
        )


# ============================================================
# LEGACY LOCAL CATALOGUE IMAGE ROUTE
# Retained for backwards compatibility.
# ============================================================


@app.route(
    "/catalogue/<domain>/<filename>",
    methods=["GET"],
)
def catalogue_image(domain, filename):
    if domain not in {"gold", "prototypes"}:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Invalid domain.",
                }
            ),
            400,
        )

    directory = MATCHING_DATA_DIR / domain

    return send_from_directory(directory, filename)


# ============================================================
# LOCAL DEVELOPMENT ONLY
# Render should use a production WSGI server.
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", 5000)),
        debug=False,
    )
