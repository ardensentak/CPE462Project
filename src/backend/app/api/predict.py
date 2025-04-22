import tempfile
from fastapi import APIRouter, UploadFile, File
from app.model.classifier import classify_image
import os

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = classify_image(tmp_path)
        return {"prediction": result}

    except Exception as e:
        print("Backend Error:", e)
        return {"error": str(e)}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

