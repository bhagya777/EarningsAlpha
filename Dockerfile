# STAGE 1: BUILDER (Compiles libraries)
# Use 'bullseye' full image to ensure C++ compilers (gcc) are present
FROM python:3.9-bullseye AS builder
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch FIRST
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install transformers \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# STAGE 2: RUNNER (Slim, Production Image)
FROM python:3.9-slim-bullseye
WORKDIR /app

# Installing runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# COPYING the compiled libraries from the builder stage
# This moves them from the temporary folder to the system Python path
COPY --from=builder /install /usr/local

# Copying application code
COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]