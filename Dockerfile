# TurboLlama Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TURBOLLAMA_CACHE_DIR=/app/models
ENV TURBOLLAMA_CONFIG_DIR=/app/config

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install TurboLlama
RUN pip install -e .

# Create directories
RUN mkdir -p /app/models /app/config /app/logs

# Copy default configuration
COPY config/ /app/config/

# Expose ports
EXPOSE 11434 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/health || exit 1

# Default command
CMD ["turbollama", "serve", "--host", "0.0.0.0", "--port", "11434", "--gui", "--gui-port", "7860"]
