# 1. Install and Import necessary libraries
import os
import zipfile
import gdown
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 2. Download and Unzip the dataset
# Using the File ID from your link
file_id = '1219EeGE1XTJVXYaulynJSa3BXGsbNCLx'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'pneumonia_dataset.zip'

print("Downloading dataset...")
gdown.download(url, output, quiet=False)

print("Unzipping dataset...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('dataset_content')

# Define directory paths
# Based on your description: _MACOSX, test, train, val are at the root
base_path = 'dataset_content'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
val_dir = os.path.join(base_path, 'val')

# 3. Data Preprocessing & Augmentation
# Images are resized to 150x150 and pixel values normalized to [0, 1]
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 4. Building the Convolutional Neural Network (CNN)
model = models.Sequential([
    # First Convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    # Second Convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Third Convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Flattening and Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting
    layers.Dense(1, activation='sigmoid') # Binary output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Training the Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 6. Evaluation on Test Data
test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")
