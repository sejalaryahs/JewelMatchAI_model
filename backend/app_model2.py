# from io import BytesIO

# from bson import ObjectId
# from flask import Flask, jsonify, request, send_file
# from flask_cors import CORS
# from gridfs import GridFS
# from PIL import Image, UnidentifiedImageError

# from .database.mongodb import (
#     get_database,
#     test_mongodb_connection,
# )
# from .onnx_matcher import ONNXJewelleryMatcher, MODEL_ID

# # ============================================================
# # FLASK APPLICATION
# # ============================================================

# app = Flask(__name__)

# CORS(app)

# # Maximum uploaded image size: 10 MB
# app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# # Load Model 2 ONNX matcher once when the worker starts.
# matcher = ONNXJewelleryMatcher()

# app.logger.info("[MODEL 2] Flask application initialized")
# app.logger.info("[MODEL 2] Model ID: %s", MODEL_ID)


# # ============================================================
# # ERROR HANDLERS
# # ============================================================


# @app.errorhandler(413)
# def request_too_large(_error):
#     app.logger.warning("[MODEL 2] Upload rejected: file exceeds 10 MB")

#     return (
#         jsonify(
#             {
#                 "status": "error",
#                 "message": "Image is too large. Maximum upload size is 10 MB.",
#             }
#         ),
#         413,
#     )


# @app.errorhandler(404)
# def not_found(_error):
#     return (
#         jsonify(
#             {
#                 "status": "error",
#                 "message": "The requested endpoint was not found.",
#             }
#         ),
#         404,
#     )


# @app.errorhandler(500)
# def internal_server_error(_error):
#     app.logger.exception("[MODEL 2] Unhandled application error")

#     return (
#         jsonify(
#             {
#                 "status": "error",
#                 "message": "Internal server error. Check the service logs.",
#             }
#         ),
#         500,
#     )


# # ============================================================
# # ROOT
# # ============================================================


# @app.get("/")
# def index():
#     return jsonify(
#         {
#             "service": "JewelMatchAI Model 2",
#             "model": "Segmented DINOv2 ONNX",
#             "model_id": MODEL_ID,
#             "status": "running",
#         }
#     )


# # ============================================================
# # HEALTH CHECK
# # ============================================================


# @app.get("/api/health")
# def health():
#     try:
#         database_ok = bool(test_mongodb_connection())

#     except Exception:
#         app.logger.exception("[MODEL 2 HEALTH] MongoDB health check failed")

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "model": "segmented_dinov2_onnx",
#                     "model_id": MODEL_ID,
#                     "mongodb": "unavailable",
#                 }
#             ),
#             503,
#         )

#     if not database_ok:
#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "model": "segmented_dinov2_onnx",
#                     "model_id": MODEL_ID,
#                     "mongodb": "unavailable",
#                 }
#             ),
#             503,
#         )

#     return jsonify(
#         {
#             "status": "ok",
#             "model": "segmented_dinov2_onnx",
#             "model_id": MODEL_ID,
#             "mongodb": "connected",
#             "provider": "CPUExecutionProvider",
#         }
#     )


# # ============================================================
# # JEWELLERY MATCHING
# # ============================================================


# @app.route("/api/match/experimental", methods=["POST"])
# @app.route("/api/match", methods=["POST"])
# def match_jewellery():
#     """
#     Multipart/form-data fields:

#         image:  Uploaded jewellery image (required)
#         source: gold or prototype (optional)
#         top_k:  Number of results (optional)

#     source='gold':
#         Search prototype catalogue.

#     source='prototype':
#         Search gold catalogue.
#     """

#     app.logger.info("[MODEL 2 POST] Request received")
#     app.logger.info(
#         "[MODEL 2 POST] Content length: %s",
#         request.content_length,
#     )

#     # --------------------------------------------------------
#     # VALIDATE UPLOADED FILE
#     # --------------------------------------------------------

#     if "image" not in request.files:
#         app.logger.warning("[MODEL 2 POST] Missing image field")

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "Upload an image using the 'image' field.",
#                 }
#             ),
#             400,
#         )

#     uploaded_file = request.files["image"]

#     if not uploaded_file.filename:
#         app.logger.warning("[MODEL 2 POST] Empty image filename")

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "No image file was selected.",
#                 }
#             ),
#             400,
#         )

#     source = (
#         request.form.get(
#             "source",
#             "prototype",
#         )
#         .strip()
#         .lower()
#     )

#     top_k = request.form.get("top_k", 5)

#     if source not in ("gold", "prototype"):
#         app.logger.warning(
#             "[MODEL 2 POST] Invalid source: %s",
#             source,
#         )

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "source must be 'gold' or 'prototype'.",
#                 }
#             ),
#             400,
#         )

#     # --------------------------------------------------------
#     # OPEN IMAGE AND RUN MATCHING
#     # --------------------------------------------------------

#     try:
#         app.logger.info(
#             "[MODEL 2 POST] Image received: %s",
#             uploaded_file.filename,
#         )

#         with Image.open(uploaded_file.stream) as opened_image:
#             image = opened_image.convert("RGB")

#         app.logger.info(
#             "[MODEL 2 POST] Image opened successfully: size=%s",
#             image.size,
#         )

#         app.logger.info(
#             "[MODEL 2 POST] Starting matching: source=%s, top_k=%s",
#             source,
#             top_k,
#         )

#         matches = matcher.find_matches(
#             image=image,
#             source=source,
#             top_k=top_k,
#         )

#         target = "prototype" if source == "gold" else "gold"

#         app.logger.info(
#             "[MODEL 2 POST] Matching completed: source=%s, " "target=%s, results=%s",
#             source,
#             target,
#             len(matches),
#         )

#         # Do not expose internal similarity scores to the frontend.
#         public_matches = []

#         for match in matches:
#             public_match = {
#                 key: value for key, value in match.items() if key != "score"
#             }
#             public_matches.append(public_match)

#         return (
#             jsonify(
#                 {
#                     "status": "success",
#                     "model_type": "segmented_dinov2_onnx",
#                     "model_id": MODEL_ID,
#                     "source": source,
#                     "target": target,
#                     "count": len(public_matches),
#                     "matches": public_matches,
#                 }
#             ),
#             200,
#         )

#     except UnidentifiedImageError:
#         app.logger.warning(
#             "[MODEL 2 POST] Uploaded file is not a valid image",
#         )

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "The uploaded file is not a valid image.",
#                 }
#             ),
#             400,
#         )

#     except ValueError as exc:
#         app.logger.exception("[MODEL 2 POST] Validation or matching ValueError")

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": str(exc),
#                 }
#             ),
#             400,
#         )

#     except Exception:
#         app.logger.exception("[MODEL 2 POST] Matching failed")

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "Matching failed. Check the service logs.",
#                 }
#             ),
#             500,
#         )


# # ============================================================
# # SERVE CATALOGUE IMAGES FROM GRIDFS
# # ============================================================


# @app.get("/catalogue-image/stored/<gridfs_id>")
# def stored_catalogue_image(gridfs_id):
#     """Serve a catalogue image stored in MongoDB GridFS."""

#     try:
#         object_id = ObjectId(gridfs_id)

#         database = get_database()
#         gridfs = GridFS(database)

#         if not gridfs.exists(object_id=object_id):
#             return (
#                 jsonify(
#                     {
#                         "status": "error",
#                         "message": "Catalogue image not found.",
#                     }
#                 ),
#                 404,
#             )

#         stored_file = gridfs.get(object_id)
#         image_bytes = stored_file.read()

#         return send_file(
#             BytesIO(image_bytes),
#             mimetype=stored_file.content_type or "image/jpeg",
#             download_name=stored_file.filename or "catalogue-image.jpg",
#             max_age=3600,
#         )

#     except Exception:
#         app.logger.exception(
#             "[MODEL 2 IMAGE] Could not retrieve GridFS image",
#         )

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "Could not retrieve the catalogue image.",
#                 }
#             ),
#             500,
#         )


# # ============================================================
# # CLEAR MODEL 2 EMBEDDING CACHE
# # ============================================================


# @app.post("/api/model2/cache/clear")
# def clear_model2_cache():
#     """
#     Clear cached catalogue embeddings after updating them.

#     Protect this endpoint before exposing it publicly.
#     """

#     try:
#         matcher.clear_catalogue_cache()

#         app.logger.info("[MODEL 2] Catalogue cache cleared")

#         return jsonify(
#             {
#                 "status": "success",
#                 "message": "Model 2 catalogue embedding cache cleared.",
#             }
#         )

#     except Exception:
#         app.logger.exception("[MODEL 2] Cache clearing failed")

#         return (
#             jsonify(
#                 {
#                     "status": "error",
#                     "message": "Could not clear the Model 2 cache.",
#                 }
#             ),
#             500,
#         )


# # ============================================================
# # LOCAL DEVELOPMENT
# # ============================================================

# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=False,
#     )

# #  #################################################
# #  #################################################

# import gc
# from io import BytesIO

# from bson import ObjectId
# from flask import Flask, jsonify, request, send_file
# from flask_cors import CORS
# from gridfs import GridFS
# from PIL import Image, UnidentifiedImageError

# from .database.mongodb import (
#     get_database,
#     test_mongodb_connection,
# )
# from .onnx_matcher import ONNXJewelleryMatcher, MODEL_ID


# # ============================================================
# # FLASK APPLICATION
# # ============================================================

# app = Flask(__name__)
# CORS(app)

# # Maximum uploaded image size: 10 MB.
# app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# # Load the model once per Gunicorn worker.
# matcher = ONNXJewelleryMatcher()

# app.logger.info("[MODEL 2] Flask application initialized")
# app.logger.info("[MODEL 2] Model ID: %s", MODEL_ID)


# # ============================================================
# # ERROR HANDLERS
# # ============================================================

# @app.errorhandler(413)
# def request_too_large(_error):
#     app.logger.warning("[MODEL 2] Upload rejected: file exceeds 10 MB")

#     return jsonify({
#         "status": "error",
#         "message": "Image is too large. Maximum upload size is 10 MB.",
#     }), 413


# @app.errorhandler(404)
# def not_found(_error):
#     return jsonify({
#         "status": "error",
#         "message": "The requested endpoint was not found.",
#     }), 404


# @app.errorhandler(500)
# def internal_server_error(error):
#     app.logger.error(
#         "[MODEL 2] Unhandled application error: %s",
#         error,
#         exc_info=True,
#     )

#     return jsonify({
#         "status": "error",
#         "message": "Internal server error. Check the service logs.",
#     }), 500


# # ============================================================
# # ROOT
# # ============================================================

# @app.get("/")
# def index():
#     return jsonify({
#         "service": "JewelMatchAI Model 2",
#         "model": "Segmented DINOv2 ONNX",
#         "model_id": MODEL_ID,
#         "status": "running",
#     })


# # ============================================================
# # HEALTH CHECK
# # ============================================================

# @app.get("/api/health")
# def health():
#     try:
#         database_ok = bool(test_mongodb_connection())

#     except Exception:
#         app.logger.exception("[MODEL 2 HEALTH] MongoDB health check failed")

#         return jsonify({
#             "status": "error",
#             "model": "segmented_dinov2_onnx",
#             "model_id": MODEL_ID,
#             "mongodb": "unavailable",
#         }), 503

#     if not database_ok:
#         return jsonify({
#             "status": "error",
#             "model": "segmented_dinov2_onnx",
#             "model_id": MODEL_ID,
#             "mongodb": "unavailable",
#         }), 503

#     return jsonify({
#         "status": "ok",
#         "model": "segmented_dinov2_onnx",
#         "model_id": MODEL_ID,
#         "mongodb": "connected",
#         "provider": "CPUExecutionProvider",
#     })


# # ============================================================
# # JEWELLERY MATCHING
# # ============================================================

# @app.route("/api/match/experimental", methods=["POST"])
# @app.route("/api/match", methods=["POST"])
# def match_jewellery():
#     """
#     Multipart form fields:
#         image:  Uploaded image (required)
#         source: gold or prototype (optional)
#         top_k:  Number of matches (optional)

#     gold      -> search prototype catalogue
#     prototype -> search gold catalogue
#     """
#     image = None

#     app.logger.info("[MODEL 2 POST] Request received")
#     app.logger.info(
#         "[MODEL 2 POST] Content length: %s",
#         request.content_length,
#     )

#     try:
#         if "image" not in request.files:
#             return jsonify({
#                 "status": "error",
#                 "message": "Upload an image using the 'image' field.",
#             }), 400

#         uploaded_file = request.files["image"]

#         if not uploaded_file.filename:
#             return jsonify({
#                 "status": "error",
#                 "message": "No image file was selected.",
#             }), 400

#         source = request.form.get("source", "prototype").strip().lower()
#         top_k = request.form.get("top_k", 5)

#         if source not in ("gold", "prototype"):
#             return jsonify({
#                 "status": "error",
#                 "message": "source must be 'gold' or 'prototype'.",
#             }), 400

#         app.logger.info(
#             "[MODEL 2 POST] Image received: %s",
#             uploaded_file.filename,
#         )

#         with Image.open(uploaded_file.stream) as opened_image:
#             # Validate image dimensions before creating the RGB copy.
#             if opened_image.width <= 0 or opened_image.height <= 0:
#                 return jsonify({
#                     "status": "error",
#                     "message": "Image has invalid dimensions.",
#                 }), 400

#             image = opened_image.convert("RGB")

#         app.logger.info(
#             "[MODEL 2 POST] Image opened: size=%s",
#             image.size,
#         )

#         app.logger.info(
#             "[MODEL 2 POST] Starting matching: source=%s, top_k=%s",
#             source,
#             top_k,
#         )

#         matches = matcher.find_matches(
#             image=image,
#             source=source,
#             top_k=top_k,
#         )

#         target = "prototype" if source == "gold" else "gold"

#         # Strip internal similarity scores from public API results.
#         public_matches = [
#             {
#                 key: value
#                 for key, value in match.items()
#                 if key != "score"
#             }
#             for match in matches
#         ]

#         app.logger.info(
#             "[MODEL 2 POST] Matching completed: source=%s, "
#             "target=%s, results=%s",
#             source,
#             target,
#             len(public_matches),
#         )

#         return jsonify({
#             "status": "success",
#             "success": True,
#             "model_type": "segmented_dinov2_onnx",
#             "model_id": MODEL_ID,
#             "source": source,
#             "target": target,
#             "count": len(public_matches),
#             "matches": public_matches,
#             "results": public_matches,
#         }), 200

#     except UnidentifiedImageError:
#         app.logger.warning("[MODEL 2 POST] Invalid image upload")

#         return jsonify({
#             "status": "error",
#             "message": "The uploaded file is not a valid image.",
#         }), 400

#     except ValueError as exc:
#         app.logger.warning(
#             "[MODEL 2 POST] Validation error: %s",
#             exc,
#         )

#         return jsonify({
#             "status": "error",
#             "message": str(exc),
#         }), 400

#     except Exception:
#         app.logger.exception("[MODEL 2 POST] Matching failed")

#         return jsonify({
#             "status": "error",
#             "message": "Matching failed. Check the service logs.",
#         }), 500

#     finally:
#         if image is not None:
#             image.close()

#         gc.collect()


# # ============================================================
# # SERVE CATALOGUE IMAGES FROM GRIDFS
# # ============================================================

# @app.get("/catalogue-image/stored/<gridfs_id>")
# def stored_catalogue_image(gridfs_id):
#     """Serve a catalogue image stored in MongoDB GridFS."""
#     try:
#         object_id = ObjectId(gridfs_id)
#         database = get_database()
#         gridfs = GridFS(database)

#         if not gridfs.exists(object_id=object_id):
#             return jsonify({
#                 "status": "error",
#                 "message": "Catalogue image not found.",
#             }), 404

#         stored_file = gridfs.get(object_id)
#         image_bytes = stored_file.read()

#         return send_file(
#             BytesIO(image_bytes),
#             mimetype=stored_file.content_type or "image/jpeg",
#             download_name=stored_file.filename or "catalogue-image.jpg",
#             max_age=3600,
#         )

#     except Exception:
#         app.logger.exception("[MODEL 2 IMAGE] GridFS image retrieval failed")

#         return jsonify({
#             "status": "error",
#             "message": "Could not retrieve the catalogue image.",
#         }), 500


# # ============================================================
# # CLEAR MODEL 2 EMBEDDING CACHE
# # ============================================================

# @app.post("/api/model2/cache/clear")
# def clear_model2_cache():
#     """Clear cached catalogue embeddings."""
#     try:
#         matcher.clear_catalogue_cache()
#         gc.collect()

#         app.logger.info("[MODEL 2] Catalogue cache cleared")

#         return jsonify({
#             "status": "success",
#             "message": "Model 2 catalogue embedding cache cleared.",
#         })

#     except Exception:
#         app.logger.exception("[MODEL 2] Cache clearing failed")

#         return jsonify({
#             "status": "error",
#             "message": "Could not clear the Model 2 cache.",
#         }), 500


# # ============================================================
# # LOCAL DEVELOPMENT
# # ============================================================

# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=False,
#     )

####################################################################
####################################################################

import gc
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

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)

# Maximum uploaded image size: 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load the model once per Gunicorn worker.
matcher = ONNXJewelleryMatcher()

app.logger.info("[MODEL 2] Flask application initialized")
app.logger.info("[MODEL 2] Model ID: %s", MODEL_ID)


# ============================================================
# ERROR HANDLERS
# ============================================================


@app.errorhandler(413)
def request_too_large(_error):
    app.logger.warning("[MODEL 2] Upload rejected: file exceeds 10 MB")

    return (
        jsonify(
            {
                "status": "error",
                "message": "Image is too large. Maximum upload size is 10 MB.",
            }
        ),
        413,
    )


@app.errorhandler(404)
def not_found(_error):
    return (
        jsonify(
            {
                "status": "error",
                "message": "The requested endpoint was not found.",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(
        "[MODEL 2] Unhandled application error: %s",
        error,
        exc_info=True,
    )

    return (
        jsonify(
            {
                "status": "error",
                "message": "Internal server error. Check the service logs.",
            }
        ),
        500,
    )


# ============================================================
# ROOT
# ============================================================


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


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/api/health")
def health():
    try:
        database_ok = bool(test_mongodb_connection())

    except Exception:
        app.logger.exception("[MODEL 2 HEALTH] MongoDB health check failed")

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


# ============================================================
# JEWELLERY MATCHING
# ============================================================


@app.route("/api/match/experimental", methods=["POST"])
@app.route("/api/match", methods=["POST"])
def match_jewellery():
    """
    Multipart form fields:
        image:  Uploaded image (required)
        source: gold or prototype (optional)
        top_k:  Number of matches (optional)

    gold      -> search prototype catalogue
    prototype -> search gold catalogue
    """
    image = None

    app.logger.info("[MODEL 2 POST] Request received")
    app.logger.info(
        "[MODEL 2 POST] Content length: %s",
        request.content_length,
    )

    try:
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

        app.logger.info(
            "[MODEL 2 POST] Image received: %s",
            uploaded_file.filename,
        )

        with Image.open(uploaded_file.stream) as opened_image:
            # Validate image dimensions before creating the RGB copy.
            if opened_image.width <= 0 or opened_image.height <= 0:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Image has invalid dimensions.",
                        }
                    ),
                    400,
                )

            image = opened_image.convert("RGB")

        app.logger.info(
            "[MODEL 2 POST] Image opened: size=%s",
            image.size,
        )

        app.logger.info(
            "[MODEL 2 POST] Starting matching: source=%s, top_k=%s",
            source,
            top_k,
        )

        matches = matcher.find_matches(
            image=image,
            source=source,
            top_k=top_k,
        )

        target = "prototype" if source == "gold" else "gold"

        # Strip internal similarity scores from public API results.
        public_matches = [
            {key: value for key, value in match.items() if key != "score"}
            for match in matches
        ]

        app.logger.info(
            "[MODEL 2 POST] Matching completed: source=%s, " "target=%s, results=%s",
            source,
            target,
            len(public_matches),
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "success": True,
                    "model_type": "segmented_dinov2_onnx",
                    "model_id": MODEL_ID,
                    "source": source,
                    "target": target,
                    "count": len(public_matches),
                    "matches": public_matches,
                    "results": public_matches,
                }
            ),
            200,
        )

    except UnidentifiedImageError:
        app.logger.warning("[MODEL 2 POST] Invalid image upload")

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
        app.logger.warning(
            "[MODEL 2 POST] Validation error: %s",
            exc,
        )

        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                }
            ),
            400,
        )

    except Exception:
        app.logger.exception("[MODEL 2 POST] Matching failed")

        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Matching failed. Check the service logs.",
                }
            ),
            500,
        )

    finally:
        if image is not None:
            image.close()

        gc.collect()


# ============================================================
# SERVE CATALOGUE IMAGES FROM GRIDFS
# ============================================================


@app.get("/catalogue-image/stored/<gridfs_id>")
def stored_catalogue_image(gridfs_id):
    """Serve a catalogue image stored in MongoDB GridFS."""
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
        app.logger.exception("[MODEL 2 IMAGE] GridFS image retrieval failed")

        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Could not retrieve the catalogue image.",
                }
            ),
            500,
        )


# ============================================================
# CLEAR MODEL 2 EMBEDDING CACHE
# ============================================================


@app.post("/api/model2/cache/clear")
def clear_model2_cache():
    """Clear cached catalogue embeddings."""
    try:
        matcher.clear_catalogue_cache()
        gc.collect()

        app.logger.info("[MODEL 2] Catalogue cache cleared")

        return jsonify(
            {
                "status": "success",
                "message": "Model 2 catalogue embedding cache cleared.",
            }
        )

    except Exception:
        app.logger.exception("[MODEL 2] Cache clearing failed")

        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Could not clear the Model 2 cache.",
                }
            ),
            500,
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
