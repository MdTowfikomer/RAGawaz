# ==============================================================================
# Dockerfile for Railway / Cloud deployment
# Index files downloaded from HuggingFace Dataset at startup
# ==============================================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code only (no data files)
COPY backend/ ./backend/
COPY app.py ./
COPY benchmark.py ./

# Create empty data directory (populated at startup)
RUN mkdir -p backend/data/multilingual_index_bundle

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    SKIP_PARENT_TEXTS=true \
    EMBEDDING_MODEL=bge_m3

EXPOSE 8080

CMD ["python", "app.py"]
