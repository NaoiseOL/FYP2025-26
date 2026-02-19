from pydantic import BaseModel, constr, conint
from datetime import datetime

class Prediction(BaseModel):
    pred_id: int
    image: bytes
    prediction: constr(min_length=1)
    datetime: datetime