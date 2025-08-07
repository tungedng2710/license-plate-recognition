import cv2
import numpy as np
# from utils.utils import COLOR_HSV

def classify_color(hsv_crop, threshold=10000):
    # Define color ranges in HSV
    color_ranges = {
        'red': [(0, 70, 50), (10, 255, 255)],
        'green': [(40, 40, 40), (70, 255, 255)],
        'blue': [(100, 150, 0), (140, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'white': [(0, 0, 200), (180, 20, 255)],
        'black': [(0, 0, 0), (180, 255, 30)],
        'gray': [(0, 0, 100), (180, 50, 200)]
    }
    # color_ranges = COLOR_HSV

    # Initialize a dictionary to store the count of pixels for each color
    color_pixel_counts = {color: 0 for color in color_ranges}

    # Iterate over the defined color ranges
    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the current color range
        lower_np = np.array(lower, dtype="uint8")
        upper_np = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv_crop, lower_np, upper_np)
        
        # Count the number of white pixels in the mask
        count = cv2.countNonZero(mask)
        color_pixel_counts[color] = count

    # Filter colors that exceed the pixel count threshold
    # max_color = max(color_pixel_counts.items(), key=lambda item: item[1])
    detected_colors = [color for color, count in color_pixel_counts.items() if count > threshold]
    detected_colors = ", ".join(detected_colors)

    return detected_colors

def recognize_color(image: np.ndarray = None):
    # Load the image
    assert image is not None
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    # contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     # x, y, w, h = cv2.boundingRect(contour)
    #     # if w > 50 and h > 50:
    car_crop = image
    hsv_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    # cv2.imwrite("temp.jpg", hsv_crop)
    car_color = classify_color(hsv_crop)
    return car_color

if __name__ == "__main__":
    recognize_color()
