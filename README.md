# DTD Texture Classification using EfficientNetB0

A Computer Vision project for multi-class texture classification using the **Describable Textures Dataset (DTD)** and an **EfficientNetB0-based transfer learning pipeline**.

## Project Overview

Texture is an important visual characteristic in computer vision. In this project, the task is to classify texture images into one of the predefined texture categories provided by the Describable Textures Dataset (DTD).

The project follows an end-to-end computer vision workflow:

Dataset
→ Exploratory Data Analysis
→ Data Validation
→ Image Preprocessing
→ Label Encoding
→ Data Augmentation
→ EfficientNetB3 Transfer Learning
→ Model Training
→ Evaluation

## Dataset

### Describable Textures Dataset (DTD)

The Describable Textures Dataset is a collection of texture images "in the wild", organized into **47 human-perception-inspired texture categories**.

The dataset contains:

- 5,640 images
- 47 texture categories
- 120 images per category
- Predefined training, validation and test splits
- 40 images per class in each split

The project uses the dataset's provided split files rather than creating a new random train/validation/test split.

### Dataset Source

Describable Textures Dataset — Visual Geometry Group, University of Oxford

[[OFFICIAL DTD DATASET PAGE]](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html)

### Dataset Citation

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014).

**Describing Textures in the Wild.**

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

## Project Workflow

### 1. Dataset Configuration

The project uses:

- Image size: 224 x 224
- Batch size: 32
- Random seed: 42
- Number of classes: 47
- DTD split: Split 1

The dataset is loaded using the provided:

- `train1.txt`
- `val1.txt`
- `test1.txt`

Each split file is used to obtain the corresponding image paths and texture labels.

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

Images are prepared for the EfficientNetB3 model using the following pipeline:

1. Read image
2. Convert BGR to RGB
3. Resize to 224 x 224
4. Convert to `float32`
5. Apply EfficientNet-compatible preprocessing

The preprocessing is designed specifically for the selected EfficientNetB3 architecture.

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

