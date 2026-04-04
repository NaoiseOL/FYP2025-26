from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints, ConfigDict, EmailStr
from datetime import datetime


NameStr = Annotated[str, StringConstraints(min_length=1, max_length=100)]
PredStr = Annotated[str, StringConstraints(min_length=4, max_length=4)]
ConfStr = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class PredCreate(BaseModel):
    image_name: NameStr
    prediction: PredStr
    confidence: ConfStr
    date_time: datetime

class PredRead(BaseModel):
    pred_id: int
    image_name: NameStr
    prediction: PredStr
    confidence: ConfStr
    date_time: datetime

    model_config= ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"