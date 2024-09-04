import tensorflow as tf
import pygetwindow as gw
import numpy as np
from PIL import ImageGrab
import pyautogui
import time

# Load your trained object detection model
model = tf.keras.models.load_model('path/to/saved/model')

# Capture game window function
def capture_game_window():
    window_title = 'BeamNG.drive'
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print("Game window not found.")
        return None
    window = windows[0]
    left, top, width, height = window.left, window.top, window.width, window.height
    screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
    return np.array(screenshot)

# Function to simulate key presses
def press_key(action):
    if action == 'left':
        pyautogui.keyDown('left')
        pyautogui.keyUp('right')  # Ensure right key is released
    elif action == 'right':
        pyautogui.keyDown('right')
        pyautogui.keyUp('left')  # Ensure left key is released
    elif action == 'forward':
        pyautogui.keyDown('up')
        pyautogui.keyUp('down')  # Ensure down key is released
    elif action == 'brake':
        pyautogui.keyDown('down')
        pyautogui.keyUp('up')  # Ensure up key is released
    else:
        # If no action, release all keys
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')
        pyautogui.keyUp('up')
        pyautogui.keyUp('down')

# Function to determine action based on detections
def determine_action(detections):
    # Placeholder logic: Replace with your actual logic based on detections
    # Example: If the car is veering to the right, steer left
    action = 'forward'  # Default action is to move forward

    # Implement your decision logic based on detections here
    # Example: if detections show lane lines or road edges, decide accordingly

    return action

# Main control loop
def main():
    print("Starting AI control...")
    while True:
        screenshot = capture_game_window()
        if screenshot is not None:
            # Preprocess the screenshot for the model
            input_image = tf.image.resize(screenshot, (416, 416))  # Resize to model input size
            input_image = np.expand_dims(input_image, axis=0) / 255.0  # Normalize

            # Run object detection model
            detections = model.predict(input_image)

            # Determine action from detections
            action = determine_action(detections)

            # Perform the action
            press_key(action)

        # Add a delay to control the frequency of actions
        time.sleep(0.1)  # Adjust as needed for your hardware

if __name__ == "__main__":
    main()
