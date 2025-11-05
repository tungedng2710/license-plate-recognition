FROM paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# RUN pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install paddleocr
RUN pip install fastapi uvicorn av aiortc python-multipart
RUN pip install ultralytics easydict onnx onnxruntime-gpu protobuf filterpy ton-ocr
RUN pip install opencv-python scikit-learn scikit-image matplotlib tqdm pandas seaborn shapely pyclipper imgaug

# Copy application code
COPY . .

# Expose port for web application
EXPOSE 7867

# Default command runs the FastAPI web app
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "7867"]
