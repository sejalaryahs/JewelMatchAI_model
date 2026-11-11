
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

COPY models/jewellery_model3.onnx ./models/jewellery_model3.onnx

EXPOSE 5002

CMD ["python", "-m", "backend.app"]
