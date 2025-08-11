import os
import random
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def check_folder(path: str = None):
    """ Create folder if it doesn't exist! """
    assert path is not None
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def check_datayaml_yolov5(saved_folder: str = None):
    """
    Create data.yaml for yolov5 dataset
    """
    if not os.path.exists(os.path.join(saved_folder, "data.yaml")):
        datayaml = "path: "+saved_folder+'\n'\
                    + "train: train/images\n" \
                    + "val: valid/images\n" \
                    + '\n' \
                    + "nc: 1\n" \
                    + "names:\n" \
                    + " 0: plate"             
        saved_path = os.path.join(saved_folder, "data.yaml")
        with open(saved_path, 'w') as file:
            file.write(datayaml)

def check_datayaml_yolov7(saved_folder: str = None):
    """
    Create data.yaml for yolov7 dataset
    """
    if not os.path.exists(os.path.join(saved_folder, "data.yaml")):
        datayaml = "train: "+os.path.join(saved_folder, "train/images")+'\n' \
                    + "val: "+os.path.join(saved_folder, "valid/images")+'\n' \
                    + '\n' \
                    + "nc: 1\n" \
                    + "names: ['plate']"
        saved_path = os.path.join(saved_folder, "data.yaml")
        with open(saved_path, 'w') as file:
            file.write(datayaml)

def xywh2yolo(bbox_xywh, im_height, im_width):
    """
    Convert bounding box (xywh) to YOLO box
    Original box: (x_topleft, y_topleft, width, height)
    """
    x_center = (2 * bbox_xywh[0] + bbox_xywh[2]) / 2
    y_center = (2 * bbox_xywh[1] + bbox_xywh[3]) / 2
    yolo_box = np.zeros(4)
    yolo_box[0] = x_center / im_width
    yolo_box[1] = y_center / im_height
    yolo_box[2] = bbox_xywh[2] / im_width
    yolo_box[3] = bbox_xywh[3] / im_height
    return list(yolo_box)

class PlateYOLO(Dataset):
    def __init__(self,
                 root_dir: str = None,
                 limit_size: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225]),
            ])
        self.image_names = [name for name in os.listdir(root_dir) if ".jpg" in name]
        if limit_size:
            self.image_names = self.image_names[:1000]
        self.data = self.preload()
        data_size = len(self.data)
        train_data = self.data[:int(data_size*0.8)]
        valid_data = self.data[int(data_size*0.8)+1:]
        self.sets = {
            "training": train_data,
            "validation": valid_data
        }

    def preload(self):
        """
        Create a list of dataset dictionary
        """
        data = []
        for image_name in self.image_names:
            data_dict = {}
            txt_name = image_name.replace(".jpg", ".txt")
            with open(os.path.join(self.root_dir, txt_name)) as file:
                info = file.read().splitlines()
            bbox = info[0].split(' ')
            bbox = [int(float(e)) for e in bbox]
            data_dict["image_path"] = os.path.join(self.root_dir, image_name)
            data_dict["label"] = int(bbox[0])
            data_dict["bbox_xywh"] = bbox[1:]
            data.append(data_dict)
        return data

    def __getitem__(self, index: int = 0):
        if index >= self.__len__():
            index = random.randint(0, self.__len__()-1)
        return self.data[index]

    def visualize(self, index: int = 0):
        if index >= self.__len__():
            index = random.randint(0, self.__len__()-1)
        sample = self.data[index]
        box = sample["bbox_xywh"]
        image = cv2.imread(sample["image_path"])
        image = cv2.circle(image, (box[0], box[1]), 2, (255, 0, 0), 5)

        plt.imshow(image[:,:,::-1])
        plt.show()

    def __len__(self):
        return len(self.data)

    def save(self,
             data_format: str = "yolov7",
             save_dir: str = "plate_yolov7",
             limit_sample: bool = False):
        print("Converting dataset to {form} format".format(form=data_format))
        check_folder(save_dir)
        if "yolo" in data_format:
            # Create data folder
            if "v5" in data_format:
                check_datayaml_yolov5(save_dir)
            else:
                check_datayaml_yolov7(save_dir)
            train_images_path = os.path.join(save_dir, "train/images")
            valid_images_path = os.path.join(save_dir, "valid/images")
            train_labels_path = os.path.join(save_dir, "train/labels")
            valid_labels_path = os.path.join(save_dir, "valid/labels")
            check_folder(train_images_path)
            check_folder(valid_images_path)
            check_folder(train_labels_path)
            check_folder(valid_labels_path)
            for key in self.sets.keys():
                if "train" in key:
                    target_images_path = train_images_path
                    target_labels_path = train_labels_path
                else:
                    target_images_path = valid_images_path
                    target_labels_path = valid_labels_path
                print("Creating {name} dataset...".format(name=key))
                for sample in tqdm(self.sets[key]):
                    # copy image file to destination
                    shutil.copy(sample["image_path"], target_images_path)

                    # Create annotation of yolo bbox
                    image = cv2.imread(sample["image_path"]) # shape: (h, w, 3)
                    im_height = image.shape[0]
                    im_width = image.shape[1]
                    bbox_xywh = sample["bbox_xywh"]
                    yolo_box = xywh2yolo(bbox_xywh, im_height, im_width)
                    label = "0 "+''.join(str(element)+' ' for element in yolo_box)

                    # write annotation to file
                    txt_name = os.path.basename(sample["image_path"]).replace(".jpg", ".txt")
                    with open(os.path.join(target_labels_path, txt_name), "w") as text_file:
                        text_file.write(label)
        else:
            print("Unsupported format!")
        print("Your dataset has been saved at {path}".format(path=save_dir))

if __name__ == "__main__":
    plate_dataset = PlateYOLO(root_dir="data/full", limit_size=False)
    # plate_dataset.save(data_format="yolov5", 
    #                    save_dir="/home/tungn197-2/code/license_plates/plate_detection/data/plate_yolov5")
    index = random.randint(0, len(plate_dataset) - 1)
    index = 11459
    print(index, plate_dataset[index])
    plate_dataset.visualize(index)