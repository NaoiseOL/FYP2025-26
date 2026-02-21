from pydantic import BaseModel, constr
from datetime import datetime

class Prediction(BaseModel):
    pred_id: int
    image_name: constr(min_length=1)
    prediction: constr(min_length=1)
    datetime: datetime