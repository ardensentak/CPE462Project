#This script preprocesses the dataset, designs an image classification model
#trains the model based on the dataset, and saves the trained model so that it can be used

import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.models import Model
import numpy as np
import os

from PIL import Image
import shutil
from collections import defaultdict

#function to preprocess data using image processing techniques
def preprocess_image(img):
    img = tf.image.resize(img, [224, 224])  #resizes to 224 x 224
    img = img / 255.0 #normalizes between 0 and 1
    img = (img - 0.5) * 2.0 #scales from -1 to 1
    img = tf.image.random_brightness(img, max_delta=0.1) #randomly adjusts image brightness
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1) # randomly adjusts image contrast
    img = tf.image.random_flip_left_right(img) #randomly rotates (flips) images
    return img

#preprocceses training data and validation data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image, horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

#creates a directory for corrupted files
corrupted_files_dir = "corruptedFiles"
os.makedirs(corrupted_files_dir, exist_ok=True)

# Function to move corrupted files out of the dataset
def move_to_corrupted_files(img_path):
    try:
        file_name = os.path.basename(img_path)
        destination = os.path.join(corrupted_files_dir, file_name)
        shutil.move(img_path, destination)
        print(f"Moved corrupted image {file_name} to {corrupted_files_dir}")
    except Exception as e:
        print(f"Error moving corrupted file {img_path}: {e}")

# Function to clean the dataset of invalid image files (.jpg, .jpeg, .png are acceptable)
def clean_directory(directory):
    valid_extensions = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the file integrity
                    with Image.open(file_path) as img:
                        img.convert("RGB")  # Try to actually open and convert image
                except Exception as e:
                    print(f"Corrupted or unsupported file: {file_path} — {e}")
                    move_to_corrupted_files(file_path)


#Storing paths in variables to simplify code below
train_dir = "data/trainingData"
val_dir = "data/validationData"
corrupted_files_dir = "corruptedFiles"
os.makedirs(corrupted_files_dir, exist_ok=True)

# Cleans both training and validation datasets
clean_directory(train_dir)
clean_directory(val_dir)

#initializing image size to 224 x 224 and batch size to 32
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

#iterator for training dataset --> used to load images in batches instead of all at once
train_gen = train_datagen.flow_from_directory(
    train_dir, #path to directory
    target_size=IMAGE_SIZE, #size of image (224 x 224)
    batch_size=BATCH_SIZE, #number of images in the batch (32)
    class_mode='binary' #binary since dataset has two classes: recyclable and non_recyclable
)

#iterator for validation dataset
val_gen = val_datagen.flow_from_directory(
    val_dir, 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

#Function to return valid image counts
def count_images_per_class(directory):
    class_counts = defaultdict(int)
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_counts[class_name] += 1
    return class_counts

# Counts and displays the # of images in each class
train_counts = count_images_per_class(train_dir)
val_counts = count_images_per_class(val_dir)

print("Training data class counts:")
for label, count in train_counts.items():
    print(f"{label}: {count}")

print("\nValidation data class counts:")
for label, count in val_counts.items():
    print(f"{label}: {count}")

#TRAIN THE MODEL: 

#loading in a pretrained model for transfer learning: enhances feature learning
    #MobileNetV2 is a CNN used for extracting image features (edges, textures, objects...)
    #imagenet is a super large dataset  
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model initially so that these layers dont change

#custom layers
x = base_model.output #stores last layer of frozen base model so customization can go on top
x = GlobalAveragePooling2D()(x) #pooling layer --> averages feature maps to a single number
x = Dropout(0.2)(x) #dropout layer --> randomly drops 20% of units to help the model generalize better
x = Dense(64, activation='relu')(x) #fully connected layer w/ 64 neurons; relu used so model can learn recyclability features
output = Dense(1, activation='sigmoid')(x) #will output the probability an image is recyclable (recyclable = 1)

#combines pre-trained model and my output layers
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#trains only on my custom layers
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5  #goes through the dataset 5 times
)

# Unfreezes the last 30 layers of MobileNetV2 for fine tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower Learing Rate for fine tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

#trains again
fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=7  #goes through the dataset 7 times
)

#save trained model
model.save("models/recyclableClassifierFinalModel.keras")