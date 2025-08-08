# Copy this file into `train.sh`
export MINIO_ENDPOINT='0.0.0.0:9000'
export MINIO_ACCESS_KEY='minioadmin'
export MINIO_SECRET_KEY='minioadmin'
export MINIO_BUCKET='iva'
export MINIO_PREFIX='yolo_runs'

python train.py \
  --model-path ./weights/yolo11l.pt \
  --data-path ../../datasets/mobile_phone_v1.2/data.yaml \
  --epochs 10 \
  --batch 32 \
  --device 0 \
  --pretrained \
  --project ./runs \
  --auto-set-name