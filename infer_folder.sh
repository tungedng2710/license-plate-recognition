folder="data"
for file in "$folder"/*.mp4; do
    echo run inference on video: $file
    python pipeline.py --source $file --save
done