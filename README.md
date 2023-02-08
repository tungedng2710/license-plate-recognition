# Vietnamese license plate recognition

## Vehicle and plate detector
The detector based on the YOLOv8 model, please checkout the official repository [Ultralytics](https://github.com/ultralytics/ultralytics) to install enviroment for inference as well as training.

For instant usage, there are two trained model for both detection tasks are put in the ```weights``` folder (You may have to check the default path in code).

## Plate reader
Plate recognition (OCR) is in the prgress of development, it will be released soon. To fill the gap of the pipeline, I temporarily use a dummy model to generate a random results. Take it easy!

**Update**: [VietOCR](https://github.com/pbcquoc/vietocr) has been added. However, it has problem in inference speed. It is being placed as an Easter egg, please checkout the code.

## Usage
Run the script below to perform license plate recognition and display the output to monitor (press ```Q``` to exit the window)
```bat 
python pipeline.py
```