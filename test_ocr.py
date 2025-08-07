import os
import cv2
import time
import re
import tqdm
import json
from ton_ocr import TonOCRPipeline
from utils.utils import check_legit_plate, check_image_size

PLATE_READER = TonOCRPipeline()


def extract_plate_info(plate_image, conf_thres):
    """Read license plate information

    Args:
        plate_image (np.ndarray): image read by OpenCV Python
        conf_thres (float): confidence threshold for OCR

    Returns:
        (string, float): OCR result and average confidence
    """
    results = PLATE_READER.predict(plate_image)
    if len(results) > 0:
        plate_info = ''
        conf = []
        for result in results:
            plate_info += result.text + ' '
            conf.append(result.score)
        conf = sum(conf) / len(conf)
        plate_info = re.sub(r'[^A-Za-z0-9\-.]', '', plate_info)
        if conf > conf_thres and check_legit_plate(plate_info):
            return plate_info, conf
        else:
            return f'', 0
    else:
        return '', 0


if __name__ == '__main__':
    # root_dir = "data/logs1/plates"
    # conf_thres = 0.9
    # eslapse = []
    # results = {}
    # image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
    # images = [cv2.imread(path) for path in image_paths]
    # pbar = tqdm.tqdm(enumerate(images), total=len(images))

    # for idx, image in pbar:
    #     t0 = time.time()
    #     result = extract_plate_info(image, conf_thres)
    #     eslapse.append(time.time() - t0)
    #     if result[1] > conf_thres:
    #         results[image_paths[idx]] = {
    #             "Plate info": result[0],
    #             "Confidence": result[1]
    #         }
    # with open("./data/test_plate_result.json", 'w') as json_file:
    #     json.dump(results, json_file, indent=4, sort_keys=True, ensure_ascii=False)
    # print(f"Total time {sum(eslapse)}")
    # print(f"Average time per image {sum(eslapse) / len(eslapse)}")

    image = cv2.imread("./data/test_samples/images/biendo1.png")
    result = extract_plate_info(image)
    print(result)
