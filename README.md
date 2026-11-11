# Jewellery Image Training

Flask API for jewellery image matching using the Model 3 ONNX runtime and Supabase-backed catalogue data.

## Run the API

Install the runtime dependencies with `pip install -r requirements.txt`, configure the required Supabase and catalogue environment variables, then start the service with `python -m backend.app`.

Set `MODEL_PATH` when the ONNX model is stored outside `models/jewellery_model3.onnx`. The model file is not included in this repository branch.

## Training dependencies

Install `requirements-training.txt` in addition to the runtime dependencies before running the scripts in `training/`.
