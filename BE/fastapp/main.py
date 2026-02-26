import os
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends

from .database import engine, SessionLocal
from .models import Base, PredDB
from .schemas import PredCreate, PredRead
from BE.train_and_eval.training import LiteMHSA

app = FastAPI()
Base.metadata.drop_all(engine)
Base.metadata.create_all(bind=engine)

model = tf.keras.models.load_model(
    "BE/model/pixelProbeB1_CIFAKE_V2.keras",
    custom_objects={"LiteMHSA":LiteMHSA}
)
class_labels = ["real", "fake"]

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)

    print(f"Raw predictions: {preds}")
    print(f"Prediction shape: {preds.shape}")
    print(f"Max value: {np.max(preds[0])}")
    print(f"Argmax: {np.argmax(preds[0])}")

    predicted_class = class_labels[np.argmax(preds[0])]
    confidence = float(preds[0][np.argmax(preds[0])])
    return predicted_class, confidence


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/predictions", response_model=list[PredRead])
def get_predictions(db: Session = Depends(get_db)):
    stmt = select(PredDB).order_by(PredDB.pred_id)
    return list(db.execute(stmt).scalars())

@app.get("/api/predictions/{pred_id}")
def get_prediction(payload: PredCreate, db: Session = Depends(get_db)):
    pred = PredDB(**payload.model_dump())


@app.post("/api/uploadfile")
async def create_upload_file(file: UploadFile, db: Session = Depends(get_db)):
    try:
        file_path = f"BE/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            #return {"message": "File saved successfully"}

        pred_class, conf = classify_image(file_path)
    
        prediction = PredDB(
        image_name=file.filename,
        prediction=pred_class,
        confidence=conf,
        date_time=datetime.now()
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Prediction Complete")
    
    return prediction