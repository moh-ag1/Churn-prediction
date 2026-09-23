FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and make logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application configuration
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/artifacts/model

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src ./src

# Copy FastAPI entry point
COPY src/app/main.py ./main.py

# Copy trained MLflow model
COPY artifacts/model ./artifacts/model

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]