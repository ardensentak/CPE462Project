# This code defines a FastAPI endpoint (/predict)
# Function: accept an uploaded image file, temporarily save it, and pass it to a classifier function (classifier.py) to get a prediction. 
# After classification, it returns the result as a JSON response and deletes the temporary file to clean up.

import tempfile
from fastapi import APIRouter, UploadFile, File
from app.model.classifier import classify_image
import os

#create router object
router = APIRouter()

#predict router initialized by registering a POST route at /predict
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        #take in files that can be converted to jpg (any image files), and store them temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        #store result and return it
        result = classify_image(tmp_path)
        return {"prediction": result}

    #print error message if an error occurs
    except Exception as e:
        print("Backend Error:", e)
        return {"error": str(e)}
        
    #delete the temporary stored file 
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

