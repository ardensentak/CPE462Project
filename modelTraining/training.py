import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
#from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.models import Model
import numpy as np
import os

from PIL import Image
import shutil
from collections import defaultdict


#NEED TO: add in image counts and the clean up functions to add more originality

def preprocess_image(img):
    img = tf.image.resize(img, [224, 224]) 
    img = img / 255.0 #normalizes between 0 and 1
    img = (img - 0.5) * 2.0 #scales from -1 to 1
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_flip_left_right(img)
    return img

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image, horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

corrupted_files_dir = "corruptedFiles"
os.makedirs(corrupted_files_dir, exist_ok=True)

# Function to move corrupted files
def move_to_corrupted_files(img_path):
    try:
        file_name = os.path.basename(img_path)
        destination = os.path.join(corrupted_files_dir, file_name)
        shutil.move(img_path, destination)
        print(f"Moved corrupted image {file_name} to {corrupted_files_dir}")
    except Exception as e:
        print(f"Error moving corrupted file {img_path}: {e}")

# Function to clean directory of invalid image files (.jpg, .jpeg, .png)
def clean_directory(directory):
    valid_extensions = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # First verify the file integrity
                    with Image.open(file_path) as img:
                        img.convert("RGB")  # Try to fully open and convert image
                except Exception as e:
                    print(f"Corrupted or unsupported file: {file_path} — {e}")
                    move_to_corrupted_files(file_path)

# Paths
train_dir = "data/trainingData"
val_dir = "data/validationData"
corrupted_files_dir = "corruptedFiles"
os.makedirs(corrupted_files_dir, exist_ok=True)

# Clean both training and validation datasets
clean_directory(train_dir)
clean_directory(val_dir)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

def count_images_per_class(directory):
    class_counts = defaultdict(int)
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_counts[class_name] += 1
    return class_counts

# Count and display image numbers
train_counts = count_images_per_class(train_dir)
val_counts = count_images_per_class(val_dir)

print("Training data class counts:")
for label, count in train_counts.items():
    print(f"{label}: {count}")

print("\nValidation data class counts:")
for label, count in val_counts.items():
    print(f"{label}: {count}")



base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5  # You can adjust
)

# Unfreeze from a certain layer onwards (you can experiment)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower Learing Rate for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=7  # more if needed
)

model.save("models/recyclableClassifierFinalModel.keras")