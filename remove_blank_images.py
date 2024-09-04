import os
from PIL import Image
import numpy as np

# Directory containing the screenshots
input_dir = 'captured_images'

# Define a threshold for determining if an image is "blank" (white)
white_threshold = 240  # Adjust this value based on your needs

def is_blank_image(image_path):
    # Open the image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Check if all pixel values are above the white threshold
    white_pixels = np.all(img_array >= white_threshold, axis=-1)
    white_ratio = np.mean(white_pixels)
    
    # Determine if the image is mostly white
    return white_ratio > 0.95  # Adjust the ratio as needed

def remove_blank_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            if is_blank_image(file_path):
                print(f"Removing blank image: {filename}")
                os.remove(file_path)

if __name__ == "__main__":
    remove_blank_images(input_dir)
    print("Blank image removal process completed.")
