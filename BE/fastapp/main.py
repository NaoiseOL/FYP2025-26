import os
import shutil
from datetime import datetime
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from .schemas import Prediction

app = FastAPI()
predictions : list[Prediction] = []

@app.get("/api/predictions")
def get_predictions():
    return predictions

@app.get("/api/predictions/{pred_id}")
def get_prediction(pred_id: int):
    for p in predictions:
        if p.pred_id == pred_id:
            return p
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

@app.post("/api/uploadfile")
async def create_upload_file(file: UploadFile):
    try:
        file_path = f"BE/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            #return {"message": "File saved successfully"}
    
        prediction = Prediction(
        pred_id=1,
        image_name=file.filename,
        prediction="Pending",
        datetime=datetime.now()
        )

        predictions.append(prediction)

        return prediction

    except Exception as e:
        return {"message": e.args}