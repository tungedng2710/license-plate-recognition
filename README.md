# YOLO-DeepSort: License plate recognition
tungedng2710 | TonAI 2023
## ToDo
- [ ] Inference speed optimization
- [x] PPOCR
- [x] DeepSort Tracking 
- [x] EasyOCR
- [x] Plate detection
- [x] Vehicle detection

## Prerequisite
* Ubuntu 20.04 or later
* 3.10 >= Python version >= 3.7

## Vehicle and plate detector
The detector based on the YOLOv8 model, please checkout the official repository [Ultralytics](https://github.com/ultralytics/ultralytics) to install enviroment for inference as well as training.

For instant usage, there are two trained model for both detection tasks are put in the ```weights``` folder (You may have to check the default path in code).

## Plate reader
**Update 15th Feb 2023**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) has been integrated.

Plate recognition (OCR) is in the prgress of development, it will be released soon. To fill the gap of the pipeline, I temporarily use a dummy model to generate a random results. Take it easy!

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
Run the script below to perform license plate recognition and display the output to monitor (press ```Q``` to exit the window)
```bat 
python pipeline.py --source path/to/video
```
If you want to save the cropped plate image detected, add the argument ```--save``` to the script

For inference on all video file in a folder, run 
```bat
./infer_folder.sh
```