# YOLO-DeepSort: License plate recognition
tungedng2710 | TonAI 2023
## Prerequisite
* Ubuntu 20.04 or later
* Python 3.9

## Vehicle and plate detector
The detector based on the YOLOv8 model, please checkout the official repository [Ultralytics](https://github.com/ultralytics/ultralytics) to install enviroment for inference as well as training.

For instant usage, there are two trained model for both detection tasks are put in the ```weights``` folder (You may have to check the default path in code).

## Plate reader
Plate recognition (OCR) is in the prgress of development, it will be released soon. To fill the gap of the pipeline, I temporarily use a dummy model to generate a random results. Take it easy!

**Update**: [VietOCR](https://github.com/pbcquoc/vietocr) and [EasyOCR](https://github.com/JaidedAI/EasyOCR) has been added. However, it has problem in inference speed and accuracy. It is being placed as an Easter egg, please checkout the code.

## Usage
Clone this repository
```bat
git clone https://github.com/tungedng2710/license-plate-recognition.git
cd license-plate-recognition
```
Install required libraries
```bat
pip install -r requirements.txt
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