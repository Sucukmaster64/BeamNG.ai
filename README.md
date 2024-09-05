# Autonomous Driving in BeamNG with Real-Time Object Detection

⚠️⚠️This Repository is still under Development. What you are seeing is not final. Changes will be made regarding Code, directories and etc. This repository is in not even in the Alpha stage of development. The Ai Model for the first tests will be trained soon!⚠️⚠️

This project aims to develop an autonomous driving system in BeamNG.drive, leveraging real-time object detection for basic driving tasks, such as lane following, steering, accelerating, and braking. The project uses screenshots captured from the game to train a deep learning model that can detect road lines and control a vehicle in real-time using keypresses.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Data Collection](#data-collection)
- [Training the Object Detection Model](#training-the-object-detection-model)
- [Real-Time Vehicle Control](#real-time-vehicle-control)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is divided into three main stages:

1. **Data Collection**: Capture images from BeamNG.drive to create a dataset for training an object detection model.
2. **Model Training**: Train a deep learning model using the collected images to detect street lines and road elements.
3. **Real-Time Vehicle Control**: Utilize the trained model to control the vehicle in BeamNG.drive via keypresses, based on detected objects in the game environment.

## Setup and Installation

### Prerequisites

- **BeamNG.drive**: Ensure BeamNG.drive is installed on your system.
- **Python 3.9** or higher: Required for running scripts.
- **Virtual Environment**: Recommended for managing dependencies.
- **NVIDIA GPU**: Optional, but recommended for training the object detection model.

### Installation Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/beamng-autonomous-driving.git
   cd beamng-autonomous-driving
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Detectron2**:

   Follow the official [Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) or use:

   ```bash
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
   ```

5. **Additional Dependencies**:

   Install required system libraries:

   ```bash
   sudo apt update
   sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
   ```

## Data Collection

To collect data for training:

1. **Run the Data Collection Script**:

   ```bash
   python data_collection.py
   ```
   Before running the script, ensure you have a empty directory called `captured_images` in the root folder of this project.

   This script captures screenshots from the BeamNG.drive window whenever a specific key combination is pressed, `Ctrl + Shift + s` for saving a picture every 1.5 sec,  `Ctrl + Shift + r` to change the number where it should continue with saving, saving images to the `captured_images` folder.
   If needed the saving interval can be changed in `data_collection.py`

3. **Organize and Annotate Images**:

   Use tools like LabelImg or automated annotation scripts to label the images with road line data.

# EVERYTHING BEYOND THIS STEP IS STILL IN WORK AND NEEDS TO BE TESTED

## Training the Object Detection Model

1. **Prepare the Dataset**:

   Ensure that the collected and annotated images are organized into a training dataset.

2. **Train the Model**:

   **THIS PART OF THE PROJECT IS UNDER DEVELOPMENT AND WILL BE ADDED LATER**
      
   Run the training script using TensorFlow:

   ```bash
   python train_model.py
   ```

   This script uses TensorFlow to train an object detection model with the prepared dataset.

2. **Evaluate and Test**:

   Test the trained model on a validation set to ensure it can accurately detect road lines.

## Real-Time Vehicle Control

1. **Run the Control Script**:

   Use the trained model to control the vehicle in BeamNG.drive:

   ```bash
   python ai_driving.py
   ```

   This script captures images from the game, uses the model to detect road lines, and sends keypresses to control the vehicle.

2. **Adjust Parameters**:

   Adjust parameters like frame rate, detection threshold, and control sensitivity in `realtime_control.py` as needed.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

