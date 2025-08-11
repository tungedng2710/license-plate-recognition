import os
import time
import json
import logging
import yaml
import glob
import shutil

from label_studio_sdk import Client
from label_studio_sdk.converter import Converter
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def choose_yolo_model(model_name: str = "yolo11n.pt"):
    allowed = {
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
        "yolov8l-seg.pt", "yolov8x-seg.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
    }
    if model_name in allowed:
        return f"weights/{model_name}"
    logging.warning("Model not found - falling back to yolo11n.pt")
    return "weights/yolo11n.pt"


def prepare_yaml_file(images_dir, classes, yaml_out):
    yaml_data = {
        "train": images_dir,
        "val": images_dir,       # quick & dirty split
        "nc": len(classes),
        "names": classes
    }
    with open(yaml_out, "w") as f:
        yaml.dump(yaml_data, f)
    logging.info(f"Data YAML written to {yaml_out}")
    return yaml_out

def export_yolo_dataset(ls_url, api_key, project_id, output_dir):
    ls = Client(url=ls_url, api_key=api_key)
    ls.check_connection()
    project = ls.get_project(project_id)
    logging.info(f"Connected to project: {project_id}")

    snap = project.export_snapshot_create(title="YOLO Export Snapshot")
    export_id = snap["id"]
    logging.info(f"Export snapshot ID: {export_id}")

    status = project.export_snapshot_status(export_id)
    while status.is_in_progress():
        logging.info("Waiting for snapshot …")
        time.sleep(1)
        status = project.export_snapshot_status(export_id)

    code, snap_path = project.export_snapshot_download(export_id, export_type="JSON")
    if code != 200:
        logging.error(f"Snapshot download failed with status {code}")
        return None

    with open(snap_path) as f:
        tasks = json.load(f)
    logging.info(f"Exported {len(tasks)} tasks")

    # convert
    conv = Converter(config=project.params["label_config"],
                     project_dir=os.path.dirname(snap_path),
                     download_resources=False)
    conv.convert_to_yolo(input_data=snap_path, output_dir=output_dir, is_dir=False)
    logging.info("Converted to YOLO format")

    # download images
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for t in tasks:
        image_url = list(t["data"].values())[0]
        if not image_url:
            continue
        for attempt in range(1, 4):
            try:
                local = get_local_path(url=image_url,
                                       hostname=ls_url,
                                       access_token=api_key,
                                       task_id=t["id"],
                                       download_resources=True)
                shutil.copy2(local, os.path.join(img_dir, os.path.basename(local)))
                break
            except Exception as e:
                logging.error(f"Task {t['id']} download error (try {attempt}): {e}")
                time.sleep(2 ** attempt)
    return output_dir


def remove_json_files_in_current_directory():
    """
    Removes all files ending with .json in the current working directory.
    """
    current_directory = os.getcwd()
    logging.info(f"Searching for .json files in: {current_directory}")

    # Use glob to find all files matching the pattern "*.json"
    # glob.glob("*.json") returns a list of filenames (relative to CWD)
    json_files = glob.glob("*.json")

    if not json_files:
        logging.info("No .json files found in the current directory.")
        return

    for file_name in json_files:
        # Construct the full absolute path for clarity and robust removal/logging
        file_path_to_remove = os.path.abspath(file_name)
        try:
            os.remove(file_path_to_remove) # or os.remove(file_name) if CWD is guaranteed
            logging.info(f"Successfully removed: {file_path_to_remove}")
        except OSError as e:
            logging.error(f"Error removing file {file_path_to_remove}: {e}")