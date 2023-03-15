folder="data/traffic_video_clean"
for file in "$folder"/*.mp4; do
    echo run inference on video: $file
    CUDA_VISIBLE_DEVICES=1 python pipeline.py \
        --vehicle_weight weights/vehicle_yolov8n_1088.pt \
        --plate_weight weights/plate_yolov8n.pt \
        --video $file \
        --vconf 0.6 \
        --pconf 0.15 \
        --ocrconf_thres 0.9 \
        --save \
        --save_dir data/log \
        --show_zoomed_plate
done