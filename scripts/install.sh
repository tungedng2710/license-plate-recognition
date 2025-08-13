conda create --name trafficcam python=3.10 -y
conda activate trafficcam
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr
pip install ultralytics, easydict, onnx, onnxruntime-gpu, protobuf, filterpy, ton-ocr