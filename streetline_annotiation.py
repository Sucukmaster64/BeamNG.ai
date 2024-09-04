import cv2
import numpy as np
import os

# Input and output directories
input_dir = "captured_images"
output_dir = "auto_annotated_images"

os.makedirs(output_dir, exist_ok=True)

def detect_lines(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

# Process each image
for image_file in os.listdir(input_dir):
    if image_file.endswith(".png") or image_file.endswith(".jpg"):
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        detect_lines(image_path, output_path)
