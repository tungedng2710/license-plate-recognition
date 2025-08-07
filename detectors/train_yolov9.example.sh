#!/bin/bash

# Train YOLOv9 models
echo "Training YOLOv9 models..."
python yolov9/train_dual.py \
  --workers 32 \
  --device 1 \
  --batch 16 \
  --data LP-11k/data.yaml \
  --img 640 \
  --cfg yolov9/models/detect/yolov9-m.yaml \
  --weights '' \
  --name yolov9-m-plate-11k \
  --hyp yolov9/data/hyps/hyp.scratch-high.yaml \
  --min-items 0 \
  --epochs 100 \
  --close-mosaic 15

# Train GELAN models
# echo "Training GELAN models..."
# python yolov9/train.py \
#   --workers 8 \
#   --device 0 \
#   --batch 32 \
#   --data yolov9/data/coco.yaml \
#   --img 640 \
#   --cfg yolov9/models/detect/gelan-c.yaml \
#   --weights '' \
#   --name gelan-c \
#   --hyp yolov9/hyp.scratch-high.yaml \
#   --min-items 0 \
#   --epochs 500 \
#   --close-mosaic 15

