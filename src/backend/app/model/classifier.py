import tensorflow as tf
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model("models/recyclableClassifierFinalModel.keras")

def classify_image(img_path):
    """Loads an image and predicts if it's recyclable or not."""
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    print(prediction)
    print(model.predict(img_array))
    return "Recyclable" if prediction > 0.5 else "Non-Recyclable"  
