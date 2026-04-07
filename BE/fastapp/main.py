import os
import shutil
import boto3
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends

from .database import engine, SessionLocal
from .models import Base, PredDB, UserDB
from .schemas import PredCreate, PredRead
from .users.user import user_router, get_current_user
from BE.train_and_eval.training import LiteMHSA

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/user")

#Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

model = tf.keras.models.load_model(
    "BE/model/best_model.keras",
    custom_objects={"LiteMHSA":LiteMHSA}
)
class_labels = ["real", "fake"]

AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
AWS_SECRET_KEY = "AWS_SECRET_KEY"
AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
AWS_REGION = "us-east-1"
BUCKET_NAME = "pixel-probe-images"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region_name=AWS_REGION
)

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
def get_predictions(db: Session = Depends(get_db),current_user: UserDB = Depends(get_current_user)):
    stmt = (
        select(PredDB)
        .where(PredDB.user_id == current_user.user_id)
        .order_by(PredDB.pred_id)
    )
    return list(db.execute(stmt).scalars())


@app.get("/api/predictions/{pred_id}")
def get_prediction(payload: PredCreate, db: Session = Depends(get_db)):
    pred = PredDB(**payload.model_dump())


@app.post("/api/uploadfile")
async def create_upload_file(file: UploadFile, db: Session = Depends(get_db), current_user: UserDB = Depends(get_current_user)):
    try:
        file_path = f"BE/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            #return {"message": "File saved successfully"}

        s3.upload_file(file_path, BUCKET_NAME, file.filename)

        pred_class, conf = classify_image(file_path)
    
        prediction = PredDB(
        image_name=file.filename,
        prediction=pred_class,
        confidence=conf,
        date_time=datetime.now(),
        user_id=current_user.user_id,
        user_name=current_user.name
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        os.remove(file_path)

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Prediction Complete")
    
    return prediction

@app.get("/testgetCurrentUser")
def test_user(current_user: UserDB = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "name": current_user.name,
        "email": current_user.email
    }
