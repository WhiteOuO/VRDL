import os
import numpy as np
import cv2
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

# Avoid garbled output
sys.stdout.reconfigure(encoding='utf-8')
# Set image size
IMG_SIZE = (128, 128)

# Load and preprocess images
def load_images_from_folder(folder):
    images = []
    image_paths = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype('float32') / 255.0  # Normalize
            images.append(img)
            image_paths.append(img_path)
    return np.array(images), image_paths

# Build Autoencoder model
def build_autoencoder():
    input_img = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Set train and val folders
train_root = "data/train"
val_root = "data/val"

# Automatically get all category names in `train/`
for class_name in os.listdir(train_root):
    train_folder = os.path.join(train_root, class_name)
    val_folder = os.path.join(val_root, class_name)

    if not os.path.isdir(train_folder) or not os.path.isdir(val_folder):
        print(f" Category {class_name} missing `train` or `val` folder, skipping...")
        continue

    print(f" Starting to process category {class_name}...")

    # Load `train set` & `validation set`
    train_images, train_image_paths = load_images_from_folder(train_folder)
    val_images, val_image_paths = load_images_from_folder(val_folder)

    if len(train_images) == 0 or len(val_images) == 0:
        print(f" Category {class_name} has no images in `train` or `val` folder, skipping...")
        continue

    # Train Autoencoder
    autoencoder = build_autoencoder()
    autoencoder.fit(train_images, train_images, epochs=30, batch_size=32, shuffle=True, validation_data=(val_images, val_images))

    # Test `train set`
    reconstructed_train = autoencoder.predict(train_images)
    train_errors = np.mean((train_images - reconstructed_train) ** 2, axis=(1, 2, 3))

    # Test `validation set`
    reconstructed_val = autoencoder.predict(val_images)
    val_errors = np.mean((val_images - reconstructed_val) ** 2, axis=(1, 2, 3))

    # Set initial Outlier threshold
    threshold = np.percentile(train_errors, 95)
    print(f" Category {class_name} initial Outlier detection threshold: {threshold:.5f}")

    # Check `validation set` max error
    max_val_error = np.max(val_errors)
    print(f" Category {class_name} `validation set` max error: {max_val_error:.5f}")

    # If `val set` max error exceeds the threshold, relax the threshold
    if max_val_error > threshold:
        print(f" Category {class_name} `validation set` error exceeds threshold, adjusting threshold...")
        threshold = max_val_error * 1.1  # Relax by 10%
        print(f" Category {class_name} updated Outlier threshold: {threshold:.5f}")

    # **Filter Outliers in `train set` and delete them**
    num_deleted = 0
    for i, error in enumerate(train_errors):
        if error > threshold:  # Outlier if error exceeds threshold
            os.remove(train_image_paths[i])  # **Delete the image**
            num_deleted += 1
            print(f" Category {class_name} deleted `train set` Outlier: {train_image_paths[i]}")

    print(f" Category {class_name} Outlier detection complete, {num_deleted} images deleted")

print(" All categories' Outlier detection completed!")
