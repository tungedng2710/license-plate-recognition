import os
import cv2
import re
# from ton_ocr import TonOCRPipeline
from paddleocr import PaddleOCR
from utils.utils import check_legit_plate

# PLATE_READER = TonOCRPipeline()
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)


def extract_plate_info(plate_image, conf_thres=0.5):
    """Read license plate information

    Args:
        plate_image (np.ndarray): image read by OpenCV Python
        conf_thres (float): confidence threshold for OCR

    Returns:
        (string, float): OCR result and average confidence
    """
    # results = PLATE_READER.predict(plate_image)
    results = ocr.predict(input=plate_image)
    if len(results) > 0:

        plate_info = ''
        plate_info = " ".join(results[0]['rec_texts'])
        conf = sum(results[0]['rec_scores']) / len(results[0]['rec_scores']) if results[0]['rec_scores'] else float('nan')
        plate_info = re.sub(r'[^A-Za-z0-9\-.]', '', plate_info)
        if plate_info and plate_info[0].isalpha():  # starts with a letter
            if len(plate_info) > 2:
                if plate_info[2] == 'C':  # replace 'C' with 0
                    plate_info = plate_info[:2] + '0' + plate_info[3:]
        if conf > conf_thres and check_legit_plate(plate_info):
            return plate_info, conf
        else:
            return f'', 0
    else:
        return '', 0


if __name__ == '__main__':
    image = cv2.imread("./data/test_samples/bien1.png")
    result = extract_plate_info(image)
    print(result)