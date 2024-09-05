import pyautogui
import keyboard
import time
import os
import pygetwindow as gw
from PIL import ImageGrab

# Directory to save the screenshots
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)

# Flag to control the capturing state
capturing = False
start_number = 1

# Set the key combination to start/stop capturing
toggle_key = 'ctrl+shift+s'  # Example: 'ctrl+shift+s' to toggle capturing
resume_key = 'ctrl+shift+r'  # Example: 'ctrl+shift+r' to resume with a starting number

def toggle_capturing():
    global capturing
    capturing = not capturing
    if capturing:
        print(f"Started capturing screenshots from {start_number}.")
    else:
        print("Paused capturing screenshots.")

def set_start_number():
    global start_number
    try:
        start_number = int(input("Enter the number to start from: "))
        print(f"Resuming capturing from {start_number}.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Bind the key combinations
keyboard.add_hotkey(toggle_key, toggle_capturing)
keyboard.add_hotkey(resume_key, set_start_number)

def get_game_window():
    # Replace 'BeamNG.drive' with the exact title of the BeamNG window if different
    window_title = 'BeamNG.drive - 0.32.5.0.16716 - RELEASE - Direct3D11'
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print("Game window not found. Ensure BeamNG.drive is running.")
        return None
    return windows[0]  # Assuming the first window with the title is the correct one

def capture_screenshot(window, number):
    # Get the window's dimensions
    left, top, width, height = window.left, window.top, window.width, window.height

    # Capture the screenshot of the specified window
    screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
    screenshot_path = os.path.join(output_dir, f"{number}.png")
    screenshot.save(screenshot_path)
    print(f"Saved screenshot to {screenshot_path}")

try:
    print(f"Press {toggle_key} to start/stop capturing screenshots.")
    print(f"Press {resume_key} to set the starting number.")
    window = None

    while True:
        if capturing:
            if window is None:
                window = get_game_window()
                if window is None:
                    time.sleep(1)
                    continue

            # Capture screenshot
            capture_screenshot(window, start_number)
            start_number += 1

            # Wait for 1 or 2 seconds
            time.sleep(1.5)  # Adjust to 2 if needed
        else:
            # Sleep to reduce CPU usage when not capturing
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Script terminated by user.")
finally:
    print("Exiting...")
