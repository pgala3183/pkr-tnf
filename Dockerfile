FROM python:3.11-slim

WORKDIR /app

# System deps (minimal; onnxruntime wheels are self-contained on manylinux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY configs/ configs/
COPY data/processed/vocab.json data/processed/vocab.json
COPY src/ src/

# Pre-exported quantized model (run export_onnx.py before docker build)
COPY checkpoints/onnx/model.int8.onnx checkpoints/onnx/model.int8.onnx

RUN pip install --no-cache-dir .

ENV ONNX_MODEL_PATH=/app/checkpoints/onnx/model.int8.onnx

EXPOSE 8000

CMD ["uvicorn", "poker_transformer.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
