#Evaluates model performance based on the validation data or the training data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

# Load trainined model
model = tf.keras.models.load_model("models/recyclableClassifierFinalModel.keras")

# Load datasets
train_dir = "data/trainingData" 
val_dir = "data/validationData"

#iterable dataset objeect for training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary',
    shuffle=False,  # Important for matching y_true and predictions
    #class_names=["non_recyclable", "recyclable"]
)

#iterable dataset object for validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary',
    shuffle=False,  
)

#preprocess function to resize same as training script
#but skipping data augmentation since its just evaluating the model
def preprocess(image, label):
    image = image / 255.0
    image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
    return image, label

#preprocesses
val_ds = val_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)
#uncomment training and comment out validation if want to run it for training data
#train_ds = train_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)

# Will collect actual labels and prediction labels
y_true = []
y_pred = []

#make predictions for all the images in the validation dataset
for images, labels in val_ds: #switch to train_ds if running it for training dataset

    preds = model.predict(images) #predict is a built in function for Keras models
    preds_binary = (preds > 0.5).astype(int).flatten() #0 = non_recyclable & 1 = recyclable
    y_true.extend(labels.numpy().astype(int)) #adds actual label to true list
    y_pred.extend(preds_binary) #adds predicted label to prediction list

# Generates a Confusion Matrix (uses sklearn.metrics)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Recyclable', 'Recyclable'])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix on Validation Set") #switch to Training when running it for training dataset
plt.show()

#Prints classification results
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-Recyclable', 'Recyclable']))

