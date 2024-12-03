# license-plate-detc-ection-using-yolov8

This project demonstrates license plate detection using the YOLOv8 model. The system can identify and locate license plates in images with high accuracy, making it suitable for tasks such as vehicle monitoring, parking systems, and traffic enforcement.

Table of Contents

Overview
Features
Dataset
Model Training
Dependencies
Usage
Results
Acknowledgments
Overview

License plate detection is a critical application in the field of computer vision. This project uses the YOLOv8 (You Only Look Once, Version 8) model, known for its speed and accuracy in real-time object detection tasks.

The project includes:

Preprocessing and custom dataset preparation.
Training the YOLOv8 model on a license plate detection dataset.
Performing inference to detect license plates in test images.
Features

Real-time license plate detection.
High accuracy and speed using YOLOv8 architecture.
Custom training on labeled license plate datasets.
Dataset

The dataset used for this project is the License Plate Detector dataset from Roboflow. It contains annotated images of vehicles with visible license plates, and was split into training, validation, and test sets.

You can access and explore the dataset here:
License Plate Detector - Roboflow

Dataset Directory Structure:

License-Plate-Detector/
├── train/
├── valid/
└── test/
Model Training

The YOLOv8 model was fine-tuned on the custom license plate dataset using the following parameters:

Optimizer: SGD
Learning Rate: 0.01
Batch Size: 16
Epochs: 100
Training was performed using the Ultralytics YOLOv8 framework. Checkpoints and the best weights are saved in the runs/detect/train directory.

Dependencies

Install the required dependencies using the requirements.txt file:

pip install -r requirements.txt
Key packages include:

Ultralytics YOLOv8
PyTorch
OpenCV
Roboflow (for dataset preparation)
Usage

1. Clone the Repository
git clone https://github.com/Chaitanyakota9/License-Plate-Detection-YOLOv8.git
cd License-Plate-Detection-YOLOv8
2. Install Dependencies
pip install -r requirements.txt
3. Model Training
Train the model on the custom dataset:

from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Start with a pre-trained YOLOv8 model

# Train the model
model.train(data='data.yaml', epochs=100, batch=16, imgsz=640)
4. Inference
Run predictions on a sample image:

from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Perform inference
results = model.predict(source='/path/to/test/image.jpg', save=True)
Results

mAP50: Achieved 97.7% mean average precision (mAP) on the test set.
Example Predictions: Annotated images with detected license plates are saved in the runs/predict folder.
Acknowledgments

Ultralytics YOLOv8 for providing an excellent object detection framework.
Roboflow for dataset preparation and management.

Feel free to contribute, open issues, or create pull requests to enhance this project!
