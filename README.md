# DTD Texture Classification using EfficientNetB0

A Computer Vision project for **47-class describable texture attribute classification** using the **Describable Textures Dataset (DTD)** and an **EfficientNetB0 transfer learning pipeline**.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Acknowledgments](#acknowledgments)

---

## Overview

Texture is an important visual characteristic in Computer Vision that provides information about the appearance and structure of objects and surfaces.

This project focuses on classifying images from the **Describable Textures Dataset (DTD)** into one of **47 human-perception-inspired texture categories**, such as *knitted, cracked, scaly, veined,* and *woven*.

The project follows an end-to-end Computer Vision workflow:

**Dataset → Exploratory Data Analysis → Data Validation → Image Preprocessing → Label Encoding → Data Augmentation → Transfer Learning → Model Training → Evaluation → Prediction**

---

## Problem Statement

Unlike conventional object classification, where the primary objective is to identify *what object* appears in an image, texture classification focuses on identifying **visual texture attributes**.

For example, an image may contain a surface that can be described as:

- Knitted
- Cracked
- Scaly
- Veined
- Braided
- Woven
- Banded
- Zigzagged

The objective of this project is to develop a deep learning model capable of learning visual patterns associated with these describable texture attributes and classifying an input image into one of the 47 DTD categories.

---

## Dataset

### Describable Textures Dataset (DTD)

The **Describable Textures Dataset** is a collection of texture images "in the wild," organized into **47 human-perception-inspired texture categories**.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Images | 5,640 |
| Texture Categories | 47 |
| Images per Category | 120 |
| Training Images | 1,880 |
| Validation Images | 1,880 |
| Test Images | 1,880 |
| Images per Category per Split | 40 |
| Split Used | Split 1 |

The project uses the **predefined DTD train, validation, and test splits** rather than creating new random splits.

The following files are used to obtain the corresponding image paths and labels:

```text
train1.txt
val1.txt
test1.txt
```

### Dataset Source

Describable Textures Dataset — Visual Geometry Group, University of Oxford

[Official DTD Dataset Page](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)

### Dataset Citation

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014).

**Describing Textures in the Wild.**

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

---

## Project Workflow

### 1. Dataset Configuration

The project uses:

- Image size: 224 x 224
- Batch size: 32
- Random seed: 42
- Number of classes: 47
- DTD split: Split 1

### 2. Data Organization

The image paths and labels are organized into separate Pandas DataFrames for:

- Training
- Validation
- Testing

A combined DataFrame is also created for overall dataset inspection.

### 3. Exploratory Data Analysis

The current notebook performs the following dataset checks:

- Dataset size and class count
- Class distribution
- Missing image path verification
- Corrupt image verification
- Visual inspection of representative images
- One sample image from each texture category

These steps are performed to understand the dataset and verify its quality before model development.

### 4. Image Preprocessing

Images are prepared for the EfficientNetB0 model using the following pipeline:

1. Read image
2. Convert BGR to RGB
3. Resize to 224 x 224
4. Convert to `float32`
5. Apply EfficientNet-compatible preprocessing

The preprocessing is designed specifically for the selected EfficientNetB0 architecture.

### 5. Label Encoding

The 47 texture class names are converted into numerical class labels using `LabelEncoder`.

The encoder is fitted using the training classes and then used to transform the training, validation and test labels.

### 6. Data Augmentation

Training data augmentation is defined using:

- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast

Augmentation is intended to be applied only to training data, while validation and test data remain unchanged.

### 7. Model Development

The project uses **EfficientNetB0** as the selected convolutional neural network architecture.

The model is intended to use **transfer learning**, leveraging pretrained visual features for the 47-class texture classification task.

### 8. Model Architecture

The project uses **EfficientNetB0** with pretrained ImageNet weights as the base model for transfer learning.

The classification head consists of:

- Global Average Pooling
- Dense layer with 512 units and ReLU activation
- Batch Normalization
- Dropout with rate 0.5
- Dense layer with 256 units and ReLU activation
- Batch Normalization
- Dropout with rate 0.3
- Final Dense layer with 47 units and Softmax activation

L2 regularization is applied to the intermediate Dense layers.

The pretrained EfficientNetB0 layers are initially frozen, allowing the newly added classification layers to learn the texture-specific features.

### 9. Model Compilation

The model is compiled using:

- Optimizer: Adam
- Learning rate: 0.0003
- Loss function: Sparse Categorical Crossentropy
- Metric: Accuracy

### 10. Model Training

The model is trained using the training and validation datasets.

Training configuration:

- Batch size: 32
- Maximum epochs: 25

The following callbacks are used:

- `ModelCheckpoint` — saves the model with the best validation accuracy
- `EarlyStopping` — stops training when validation performance stops improving
- `ReduceLROnPlateau` — reduces the learning rate when validation loss stops improving

### 11. Training Performance

Training and validation accuracy and loss are monitored throughout the training process.

The model achieved approximately:

- Training accuracy: **86.44%**
- Validation accuracy: **65.11%**

Training and validation performance are visualized using accuracy and loss curves.

### 12. Model Evaluation

The trained model is evaluated using the separate DTD test dataset.

The final test performance is:

- Test Accuracy: **66.33%**
- Test Loss: **1.3648**

The test set contains 1,880 images from the 47 texture categories.

### 13. Single Image Prediction

The notebook also performs prediction on individual test images.

The prediction pipeline consists of:

1. Load the image
2. Convert BGR to RGB
3. Resize to 224 x 224
4. Apply EfficientNet preprocessing
5. Generate the prediction
6. Select the class with the highest probability
7. Display the predicted class and confidence

A sample prediction in the notebook correctly classified a `flecked` texture with a confidence of **86.38%**.

### 14. Results

The EfficientNetB0 model achieved a test accuracy of **66.33%** on the DTD test split.

The results show that transfer learning with EfficientNetB0 can effectively learn visual features for classifying the 47 different texture categories.

The difference between training and validation accuracy also indicates some degree of overfitting.

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Project Structure

```text
DTD-Texture-Classification-EfficientNetB0/
│
├── DTD_Texture_Classification.ipynb
├── README.md
├── requirements.txt
```

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DTD-Texture-Classification-EfficientNetB0.git
cd DTD-Texture-Classification-EfficientNetB0
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
tensorflow>=2.15
numpy
pandas
opencv-python
matplotlib
scikit-learn
```

### 3. Download the dataset

Download the DTD dataset from the [official page](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html) and place it so the notebook's `DATASET_PATH` points to the extracted `dtd` folder (containing `images/` and `labels/`).

### 4. Run the notebook

Open `DTD_Texture_Classification.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab, and run the cells in order — from dataset loading through training, evaluation, and prediction.

---

## Acknowledgments

- Describable Textures Dataset (DTD) — Visual Geometry Group, University of Oxford
- EfficientNetB0 pretrained weights — Keras Applications (ImageNet)
