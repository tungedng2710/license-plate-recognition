folder=data
video_name=$(ls $folder/*.mp4)
echo $video_name is detected
python pipeline.py $videos