# cnn-image-classification 
"Convolutional Neural Network (CNN) Development for Image Classification on CIFAR-10 Dataset"

This project focuses on designing, training, and evaluating a **Convolutional Neural Network (CNN)** for **multi-class image classification** using the **CIFAR-10 dataset**.  
It also explores **data augmentation techniques** to improve generalization and visualizes **CNN feature maps** to understand what the model learns at different layers.

Developed as part of my Machine Learning internship, this project enhanced my understanding of **deep learning architectures**, **computer vision preprocessing**, and **CNN interpretability**.

# Project Goals
To build and analyze a CNN model that:
- Classifies 32×32 color images into 10 object categories.
- Implements **data augmentation** to improve model robustness.
- Visualizes **feature maps** from convolutional layers for interpretability.
- Evaluates performance using **training, validation, and test accuracy**.

# Project Workflow

### 1. Dataset Loading and Preprocessing
- Loaded the **CIFAR-10 dataset** from TensorFlow/Keras datasets.
- Performed preprocessing:
  - Normalized image pixel values to a **0–1 range**.
  - One-hot encoded the labels for multi-class classification.
  - Split dataset into **training** and **testing** sets.

### 2. CNN Model Architecture Design
- Constructed a CNN using the Keras **Sequential API**, including:
  - **Convolutional layers (Conv2D)** with ReLU activations.
  - **Pooling layers (MaxPooling2D)** for dimensionality reduction.
  - **Flatten layer** to convert 2D feature maps into 1D vectors.
  - **Dense layers** for classification with a **Softmax output**.
- Compiled the model using:
  - Optimizer: `adam`
  - Loss: `categorical_crossentropy`
  - Metric: `accuracy`

### 3. Data Augmentation Implementation
- Used **ImageDataGenerator** to apply augmentation techniques:
  - Random rotations
  - Width and height shifts
  - Horizontal flips
  - (Optional) Zoom and shear transformations
- Explained benefits of augmentation:
  - Increases dataset diversity
  - Reduces overfitting
  - Improves model generalization

### 4. Model Training and Evaluation
- Trained the CNN using augmented images.
- Monitored **training and validation accuracy/loss** over epochs.
- Evaluated the trained model on the **test dataset**.
- Compared results before and after augmentation.

### 5. Feature Map Visualization
- Selected a random image from the test set.
- Visualized **feature maps** from intermediate convolutional layers.
- Analyzed how filters detect edges, colors, and patterns at different depths.

# Tools and Libraries
- **Python 3.x**
- **TensorFlow / Keras** – for CNN construction, training, and augmentation  
- **NumPy** – for numerical operations  
- **Matplotlib** – for plotting accuracy/loss and feature maps  
- **OpenCV (cv2)** or **PIL (Pillow)** – for image visualization

# Dataset Information
**Dataset:** [CIFAR-10 Image Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/krishnajaiswal8/cifar-10-dataset)  

**Description:**  
The **CIFAR-10 dataset** is one of the most widely used benchmark datasets in computer vision.  
It consists of **60,000 color images** (32×32 pixels), categorized into **10 classes**, each containing 6,000 images.

# How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cnn-image-classification.git

