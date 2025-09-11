conda create --name trafficcam python=3.11 -y
conda activate trafficcam
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr
pip install fastapi uvicorn
pip install ultralytics easydict onnx onnxruntime-gpu protobuf filterpy ton-ocr filterpy
pip install opencv-python scikit-learn scikit-image matplotlib tqdm pandas seaborn shapely pyclipper imgaug