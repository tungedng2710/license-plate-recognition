folder="data/traffic_video_clean"
for file in "$folder"/*.mp4; do
    echo run inference on video: $file
    CUDA_VISIBLE_DEVICES=1 python pipeline.py --video $file --save
done