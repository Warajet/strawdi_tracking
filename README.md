# Strawberry Tracker

## Project Description

**Strawberry Tracker** is an experiment designed to detect and count strawberries on trees using computer vision algorithms. The main objectives are to implement and train an object detector using the StrawDI dataset and to integrate this trained detector into a tracking algorithm for analyzing video sequences.

### Objectives

1. **Train Object Detector**: Implement and train an object detection model on the [StrawDI dataset](https://strawdi.github.io/).
2. **Integrate Tracking Algorithm**: Incorporate the trained object detector into a tracking algorithm to perform strawberry tracking on the provided video sequence (`test.mp4`).

## Method

### Object Detection

- **Pretrained Model**: Utilize the Faster R-CNN (Region-based Convolutional Neural Network) model as a pretrained object detector for identifying strawberries in images.
- **Training**: Fine-tune the pretrained Faster R-CNN model using the StrawDI dataset to adapt it for strawberry detection.

### Tracking Algorithm

- **Tracking Approach**: Implement tracking using the Hungarian Algorithm combined with Intersection-over-Union (IoU) metrics to track detected strawberries across video frames.

## Setup and Installation

### Prerequisites

- Python 3.x
- Required libraries: `torch`, `torchvision`, `opencv-python`, `numpy`, etc.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/strawberry-tracker.git
   cd strawdi_tracking
   ```
2. **Create a Virtual Environment**

  ```bash
  python -m venv env
  source env/bin/activate  # On Windows use `env\Scripts\activate`
  ```

3. **Install Dependencies**

  ```bash
  pip install -r requirements.txt
  ```

## Usage
### Prepare the Dataset

Download the StrawDI dataset from [here](https://drive.google.com/file/d/1elFB-q9dgPbfnleA7qIrTb96Qsli8PZl/view) and run:

  ```bash
  cd strawdi_tracking
  unzip dataset.zip
  ```
  
### Train the Object Detector

  ```bash
  python train_detector.py --dataset_path /path/to/strawdi/dataset
  ```

### Run the Tracking Algorithm

  ```bash
  python track_strawberries.py --video_path /path/to/test.mp4 --model_path /path/to/trained/model
  ```

- --video_path: Path to the video file (test.mp4).
- --model_path: Path to the trained object detection model.

## Results
Detection: The object detector should be able to identify strawberries with high accuracy.

<div align="center">
  <img src="https://github.com/Warajet/strawdi_tracking/blob/main/straw_di_detection_output.png" width="500" alt="Demo" />
</div>

Tracking: The tracking algorithm should maintain the identity of each detected strawberry throughout the video sequence.

<div align="center">
  <img src="https://github.com/Warajet/strawdi_tracking/blob/main/straw_di_tracking_output.gif" width="500" alt="Demo" />
</div>

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Contributions are always welcome!

## Contact
For any questions or further information, please contact:

- Author: Warakorn Jetlohasiri
- Email: warakornjetlohasiri@gmail.com
