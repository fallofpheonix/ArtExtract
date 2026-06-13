# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for building/installing some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies (e.g., libgomp for FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy the source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY pyproject.toml README.md ./

# Install the package itself in editable mode or just make it available via PYTHONPATH
ENV PYTHONPATH=/app/src
ENV ARTEXTRACT_OUT_DIR=/app/analysis_out

# Default command: run the similarity retrieval pipeline
ENTRYPOINT ["python", "scripts/run_similarity.py"]
CMD ["--help"]
