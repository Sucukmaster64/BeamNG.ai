import os
import cv2

# Define input and output directories
input_dir = "captured_images"
output_dir_bottom = "halved_images"
output_dir_top = "trash"

# Create output directories if they don't exist
os.makedirs(output_dir_bottom, exist_ok=True)
os.makedirs(output_dir_top, exist_ok=True)

# Process each image
for image_file in os.listdir(input_dir):
    if image_file.endswith(".png") or image_file.endswith(".jpg"):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Calculate the midpoint
        midpoint = height // 2

        # Cut the image into top and bottom halves
        top_half = image[:midpoint, :]
        bottom_half = image[midpoint:, :]

        # Save the top half to the "trash" directory
        top_half_path = os.path.join(output_dir_top, image_file)
        cv2.imwrite(top_half_path, top_half)

        # Save the bottom half to the "halved_images" directory
        bottom_half_path = os.path.join(output_dir_bottom, image_file)
        cv2.imwrite(bottom_half_path, bottom_half)

print("Image processing complete. Check the 'trash' and 'halved_images' directories.")
