# Image-Classification-Using-Tensorflow
This repository demonstrates how to build, train, and evaluate an image classification model using TensorFlow. The project includes data preprocessing, model architecture design, training, and evaluation on a sample dataset.

## Features
- Preprocess image data (resizing, normalization, and augmentation).
- Build a Convolutional Neural Network (CNN) for image classification.
- Train the model with efficient loss functions and optimizers.
- Evaluate model performance on test data.
- Save and load the trained model.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install tensorflow matplotlib numpy pandas
```

## Dataset

You can use any image classification dataset, such as:
- CIFAR-10
- Custom datasets (stored in `train`, `val`, and `test` directories).

To use a custom dataset, organize it as follows:


## Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Image-Classification-Using-Tensorflow.git
cd Image-Classification-Using-Tensorflow
```

### 2. Run the script
Train and evaluate the model:

## Code Overview

### **1. Data Preprocessing**
The dataset is preprocessed using TensorFlow's `ImageDataGenerator` or `image_dataset_from_directory` to:
- Resize images.
- Normalize pixel values to a [0, 1] range.
- Perform data augmentation (e.g., flipping, rotation).

### **2. Model Architecture**
A CNN is designed using TensorFlow's Keras API with:
- Convolutional layers for feature extraction.
- Pooling layers to reduce spatial dimensions.
- Dense layers for classification.

### **3. Training**
- Optimizer: Adam
- Loss function: Categorical Crossentropy (for multi-class classification)
- Metrics: Accuracy

### **4. Evaluation**
The model's performance is evaluated on the test dataset and visualized using confusion matrices and accuracy plots.
