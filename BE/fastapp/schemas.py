from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints, ConfigDict
from datetime import datetime

NameStr = Annotated[str, StringConstraints(min_length=1, max_length=100)]
PredStr = Annotated[str, StringConstraints(min_length=4, max_length=4)]


class PredCreate(BaseModel):
    image_name: NameStr
    prediction: PredStr
    date_time: datetime

class PredRead(BaseModel):
    pred_id: int
    image_name: NameStr
    prediction: PredStr
    date_time: datetime

    model_config= ConfigDict(from_attributes=True)