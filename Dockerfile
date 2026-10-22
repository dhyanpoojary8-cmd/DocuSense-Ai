# Use official lightweight Python 3.11 base image
FROM python:3.11-slim

# Prevent interactive prompts and Python bytecode/buffering issues
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install Tesseract OCR and English training data; clean apt cache to minimize image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with UID 1000 (required by Hugging Face Spaces Docker SDK)
RUN useradd -m -u 1000 appuser

# Set working directory inside appuser home
WORKDIR /home/appuser/app

# Copy dependency definition first for Docker layer caching
COPY --chown=appuser:appuser backend/requirements.txt ./requirements.txt

# Install dependencies into Python environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code into container
COPY --chown=appuser:appuser backend/ ./

# Ensure uploads directory exists and is owned by appuser
RUN mkdir -p uploads && chown -R appuser:appuser /home/appuser/app

# Switch to non-root user
USER appuser

# Expose the default port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Start Uvicorn bound to 0.0.0.0, dynamically using $PORT if provided (e.g. Render 10000) or defaulting to 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
