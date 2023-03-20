conda create --name trafficcam python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate trafficcam
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddleocr
pip install ultralytics
pip install easydict
pip install onnx
pip install onnxruntime-gpu==1.8
pip install protobuf==3.1.0