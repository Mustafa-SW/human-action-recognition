# Human Action Recognition using PyTorch

## Goal

Build a Human Action Recognition system using skeleton keypoints extracted from videos.

## Skills Demonstrated

- Python
- PyTorch
- Computer Vision
- Video Perception
- Linux
- Git & GitHub

## Progress

### Day 1

- Environment setup
- Installed PyTorch
- Learned tensors
- Created first PyTorch program

## Day 2

- Learned Dataset class
- Learned DataLoader
- Loaded CSV data
- Created mini-batches

## Day 3

### Video Processing with OpenCV

- Loaded and inspected video files
- Extracted video metadata (FPS, resolution, duration)
- Learned frame-by-frame video processing
- Implemented frame sampling (every 10th frame)
- Resized 4K frames for efficient processing

### Key Learnings

- Videos are sequences of images (frames)
- FPS determines temporal resolution
- Frame sampling reduces storage and computation
- OpenCV is widely used for computer vision pipelines

## Day 4

### Human Pose Estimation

- Installed MediaPipe Pose
- Detected 33 human body landmarks
- Visualized skeleton overlays on video frames
- Learned landmark coordinate representation
- Debugged and fixed MediaPipe installation issues

### Key Learnings

- Human bodies can be represented as skeletons
- Each landmark contains x, y, z coordinates
- Skeleton representations are widely used in action recognition
- Pose estimation converts visual information into numerical features

## Day 5

### Landmark Dataset Generation

- Extracted 33 MediaPipe body landmarks
- Saved pose coordinates to CSV
- Built structured dataset from video
- Learned feature extraction pipeline

### Dataset

- 33 landmarks
- 132 numerical features per frame
- Stored in CSV format

## Day 6 

## PyTorch Data Pipeline

### Goals
- Convert pose landmark CSV into PyTorch tensors.
- Build a custom Dataset class.
- Create DataLoaders for batching.

### What I Learned
- PyTorch Dataset objects manage training data.
- DataLoader automatically creates batches.
- Landmark features can be converted into tensors.
- Data inspection is important before model training.

### Key Concepts
- Dataset
- DataLoader
- Feature Vectors
- Tensor Conversion
- Batch Processing

### Results
- Created PoseDataset class.
- Loaded landmark CSV into PyTorch.
- Generated training batches of shape [8, 132].

## Day 7

### Pose Classification Model

- Built a PyTorch neural network
- Created a training pipeline
- Used DataLoader batching
- Implemented forward and backward propagation
- Saved trained model weights

### Architecture

Input: 132 pose features

132 → 64 → 32 → 2

Activation: ReLU

Optimizer: Adam

Loss Function: CrossEntropyLoss