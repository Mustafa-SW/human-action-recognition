# Learning Journal

## Day 1
- Environment setup
- Installed PyTorch
- Learned tensors

## Day 2
- Learned Dataset and DataLoader
- Built custom dataset

## Day 3
- Learned OpenCV
- Processed videos frame-by-frame
- Extracted video metadata
- Implemented frame sampling

## Day 4

- Installed MediaPipe Pose
- Learned about 33 body landmarks
- Performed pose estimation on video
- Visualized skeleton connections
- Debugged MediaPipe installation issues

## Day 5 - Landmark Dataset Generation

- Extracted 33 landmarks from sampled video frames.
- Generated a structured CSV dataset.
- Converted visual information into numerical features.
- Dataset Structure
- 33 landmarks
- 4 values per landmark: x, y, z, visibility
- Total Features per Frame: 132

## Day 6 - PyTorch Data Pipeline

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

## Day 7 - Model Training

### Goals
- Train a neural network using pose features.

### What I Learned
- Neural networks learn patterns from numerical features.
- Forward propagation generates predictions.
- Loss functions measure prediction quality.
- Backpropagation updates model weights.
- Trained models can be saved and reused.

### Key Concepts
- Neural Networks
- ReLU
- CrossEntropy Loss
- Adam Optimizer
- Model Serialization

### Results
- Built first pose classification model.
- Trained using PyTorch.
- Saved model weights to disk.