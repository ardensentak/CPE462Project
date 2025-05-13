#Note if using VScode as an IDE some imports may show up as having problems after installing requirements; 
# but its just VScode not recognizing them; the code will run as its supposed to
import tensorflow as tf
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model("models/recyclableClassifierFinalModel.keras") #load in trained model

#prediction script
def classify_image(img_path): 
    """Loads an image and predicts if it's recyclable or not."""
    img = image.load_img(img_path, target_size=(224, 224)) #load image at target size
    img_array = image.img_to_array(img) / 255.0 #normalize image pixels
    img_array = np.expand_dims(img_array, axis=0) #add axis to image array to resemble batch size (1)

    prediction = model.predict(img_array)[0][0] #predict if image is recyclable or not

    #print prediction results
    print(prediction)
    print(model.predict(img_array))
    return "Recyclable" if prediction > 0.5 else "Non-Recyclable"  
