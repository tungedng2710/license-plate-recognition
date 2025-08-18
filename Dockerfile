# Use official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for web application
EXPOSE 8000

# Default command runs the FastAPI web app
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
