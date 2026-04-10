# 🩺 Pneumonia Detection using CNN

## 📌 Overview

This project focuses on building a deep learning model to classify chest X-ray images into two categories:

* **Pneumonia**
* **Normal**

The dataset consists of 5,863 pediatric chest X-ray images collected from children aged 1–5 years. The images are divided into three sets: **train**, **validation**, and **test**.

---

## 🎯 Objective

To develop a **Convolutional Neural Network (CNN)** model that can automatically detect pneumonia from chest X-ray images with high accuracy.

---

## 📂 Dataset Structure

After extracting the dataset, the directory structure is:

dataset_content/
├── train/
│   ├── Pneumonia/
│   └── Normal/
├── val/
│   ├── Pneumonia/
│   └── Normal/
├── test/
│   ├── Pneumonia/
│   └── Normal/

---

## ⚙️ Methodology

### 🔹 1. Data Preprocessing

* Images resized to **150 × 150 pixels**
* Pixel values normalized (rescaled to 0–1)
* Data augmentation applied:

  * Shear transformation
  * Zoom
  * Horizontal flip

---

### 🔹 2. Model Architecture (CNN)

The model consists of:

* 3 Convolutional Layers (Conv2D)
* 3 MaxPooling Layers
* Flatten Layer
* Dense Layer (512 neurons)
* Dropout Layer (0.5) to reduce overfitting
* Output Layer (Sigmoid activation)

---

### 🔹 3. Training

* Optimizer: **Adam**
* Loss Function: **Binary Crossentropy**
* Epochs: **10**
* Batch Size: **32**

---

### 🔹 4. Evaluation

* Model evaluated using test dataset
* Performance metric: **Accuracy**

---

## 📊 Results

* ✅ Training Accuracy: ~90% (approx)
* ✅ Validation Accuracy: ~85% (approx)
* ✅ Test Accuracy: ~85–90% (depends on run)

The model successfully distinguishes between pneumonia and normal X-ray images.

---

## 🚀 Key Findings

* CNN performs well for medical image classification
* Data augmentation improves generalization
* Dropout layer helps reduce overfitting
* Model can be further improved using deeper architectures

---

## 🔧 Future Improvements

* Use **Transfer Learning** (VGG16, ResNet)
* Increase number of epochs
* Hyperparameter tuning
* Use larger dataset for better generalization

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 📌 Conclusion

This project demonstrates that deep learning models like CNN can effectively classify medical images and assist in early disease detection such as pneumonia.

---

