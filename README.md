# YOLO-DeepSort: License plate recognition
## ToDo
- [ ] Deployment
- [ ] ONNX Running
- [ ] Inference speed optimization
- [x] PPOCR
- [x] DeepSort Tracking 
- [x] EasyOCR
- [x] Plate Detection
- [x] Vehicle Detection

## Prerequisite
* Ubuntu 20.04 or later
* 3.10 >= Python version >= 3.7

## Overview
Both vehicle and plate detector based on the YOLOv8 model, please checkout the official repository [Ultralytics](https://github.com/ultralytics/ultralytics) to install enviroment for inference as well as training.

For instant usage, there are two trained model for both detection tasks are put in the ```weights``` folder (You may have to check the default path in code). In addition, you can train the yolov8 with your custom dataset with serveral lines of code in the ```train_yolov8.py``` file.

As regards plate recognition, [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) has been used to recognized the plate information. 

For tracking task, DeepSORT algorithm is implemented. The implementation and pretrained model are taken from [John1liu](https://github.com/John1liu/YOLOV5-DeepSORT-Vehicle-Tracking-Master).


## Usage
Clone this repository
```bat
git clone https://github.com/tungedng2710/license-plate-recognition.git
cd license-plate-recognition
```
Install required libraries
```bat
bash install.sh
```

For quick inference on all video file in a folder, run 
```bat
bash infer_folder.sh
```

The result would be saved in the directory ```data/log```, you can change the saved path by changing the ```--save_dir``` argument.

For inference on a single video, run script below
```bat
python pipeline.py --video [path_to_your_video] [optional arguments]
```
**Arguments**
- ```--video```: (str) path to video, 0 for webcam
- ```--save```: (bool) save output video
- ```--save_dir```: (str) saved path
- ```--vehicle_weight```: (str) path to the yolov8 weight of vehicle detector
- ```--plate_weight```: (str) path to the yolov8 weight of plate detector
- ```--vconf```: (float) confidence for vehicle detection
- ```--pconf```: (float) confidence for plate detection
- ```--ocrconf_thres```: (float) threshold for ocr model
- ```--stream```: (bool) real-time monitoring