import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

# Load model
model = tf.keras.models.load_model("models/recyclableClassifierFinalModel.keras")

# Load validation dataset
val_dir = "data/validationData"
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary',
    shuffle=False,  # Important for matching y_true and predictions
    #class_names=["non_recyclable", "recyclable"]
)

#preprocess same way as training
def preprocess(image, label):
    image = image / 255.0
    image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
    return image, label


val_ds = val_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)

# Collect true labels and predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds_binary = (preds > 0.5).astype(int).flatten()
    y_true.extend(labels.numpy().astype(int))
    y_pred.extend(preds_binary)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Recyclable', 'Recyclable'])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix on Validation Set")
plt.show()

# Optional: Print classification metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-Recyclable', 'Recyclable']))
