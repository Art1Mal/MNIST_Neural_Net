# MNIST_Neural_Net
"A TensorFlow-based neural network for classifying handwritten digits from the MNIST dataset."
This repository contains a TensorFlow implementation of a neural network for classifying handwritten digits (0-9) from the MNIST dataset. The project demonstrates key concepts in deep learning, such as data preprocessing, model architecture design, and regularization.

## Features

- **Data Preprocessing**: 
  - Normalization of pixel values to the range [0, 1].
  - Efficient batching, shuffling, caching, and prefetching for optimal performance.
  
- **Model Architecture**:
  - Multi-layer dense neural network with batch normalization and dropout layers.
  - L2 regularization for weight decay to prevent overfitting.
  - Softmax activation for output layer, providing probabilistic class predictions.

- **Training**:
  - Trained on the MNIST dataset for 65 epochs.
  - Sparse categorical crossentropy loss and accuracy as evaluation metrics.
  - Validation on the test dataset after each epoch.

## Dataset

The MNIST dataset is a benchmark dataset in machine learning, consisting of 70,000 grayscale images of handwritten digits (28x28 pixels). 

- **Training set**: 60,000 images
- **Testing set**: 10,000 images

The dataset is automatically downloaded and loaded using `tensorflow_datasets`.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- tensorflow-datasets

## Network Architecture

- **Layers**: 5 fully connected layers with the following configuration:
  - Layer #1: 94 neurons
  - Layer #2: 49 neurons
  - Layer #3: 81 neurons
  - Layer #4: 33 neurons
  - Output Layer: 10 classes (digits 0-9)
- Regularization: L2 weight decay
- Dropout: Applied to intermediate layers for regularization
- Optimized for different activation functions and batch sizes

## Experiments Overview

A total of **15 experiments** were conducted. Each experiment utilized different combinations of hyperparameters to evaluate their effect on training and test accuracy. 

### Best Experiment
The best performance was achieved in **Experiment #5**:
- **Hyperparameters**:
  - Activation Function: `relu`
  - Dropout Rates: (0.1, 0.2, 0.25, 0.1)
  - Batch Training Size: 48
  - Batch Testing Size: 128
  - Epochs: 65
  - Dense Regularization: (0.0001, 0.0004, 0.008, 0.003)
- **Results**:
  - Training Accuracy: 96.52%
  - Test Accuracy: 98.06%

### Summary of Experiments

| Experiment | Activation Function  | Dropout Rates            | Batch Sizes (Train/Test)| Epochs | Training Accuracy | Test Accuracy |
|------------|----------------------|--------------------------|-------------------------|--------|-------------------|---------------|
| #1         | sigmoid              | (0.1, 0.15, 0.2, 0.1)    | 32 / 64                 | 50     | 95.30%            | 97.64%        |
| #2         | sigmoid              | (0.15, 0.2, 0.3, 0.1)    | 96 / 128                | 50     | 95.85%            | 97.40%        |
| #3         | tanh                 | (0.1, 0.15, 0.2, 0.1)    | 64 / 256                | 45     | 96.50%            | 97.49%        |
| #4         | tanh                 | (0.2, 0.2, 0.25, 0.15)   | 128 / 512               | 50     | 96.56%            | 97.32%        |
| #5         | relu                 | (0.1, 0.2, 0.25, 0.1)    | 48 / 128                | 65     | **96.52%**        | **98.06%**    |
| #6         | relu                 | (0.15, 0.2, 0.3, 0.15)   | 32 / 128                | 50     | 94.51%            | 97.83%        |
| #7         | elu                  | (0.1, 0.15, 0.3, 0.1)    | 64 / 500                | 50     | 96.50%            | 97.93%        |
| #8         | elu                  | (0.15, 0.2, 0.3, 0.15)   | 128 / 256               | 50     | 97.21%            | 97.89%        |
| #9         | leaky_relu           | (0.2, 0.2, 0.25, 0.1)    | 96 / 100                | 50     | 95.69%            | 97.51%        |
| #10        | leaky_relu           | (0.15, 0.25, 0.3, 0.2)   | 32 / 64                 | 35     | 93.28%            | 97.08%        |
| #11        | sigmoid              | (0.2, 0.2, 0.3, 0.15)    | 128 / 128               | 55     | 96.28%            | 97.40%        |
| #12        | tanh                 | (0.15, 0.2, 0.2, 0.15)   | 32 / 500                | 50     | 94.29%            | 97.31%        |
| #13        | relu                 | (0.15, 0.2, 0.25, 0.1)   | 96 / 64                 | 50     | 97.06%            | 97.85%        |
| #14        | elu                  | (0.15, 0.2, 0.3, 0.2)    | 64 / 256                | 60     | 96.46%            | 97.86%        |
| #15        | leaky_relu           | (0.15, 0.2, 0.3, 0.1)    | 48 / 128                | 50     | 94.65%            | 97.04%        |
