import os
import shutil
from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    try:
        file_path = f"BE/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            return {"message": "File saved successfully"}
    except Exception as e:
        return {"message": e.args}