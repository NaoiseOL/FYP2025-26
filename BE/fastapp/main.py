import os
import shutil
from datetime import datetime
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends

from .database import engine, SessionLocal
from .models import Base, PredDB
from .schemas import PredCreate, PredRead

app = FastAPI()
Base.metadata.drop_all(engine)
Base.metadata.create_all(bind=engine)

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
    
        prediction = PredDB(
        image_name=file.filename,
        prediction="Pend",
        date_time=datetime.now()
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Prediction Complete")
    
    return prediction