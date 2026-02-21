from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime
from datetime import datetime

class Base(DeclarativeBase):
    pass

class PredDB(Base):
    __tablename__ = "predictions"

    pred_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    image_name: Mapped[str] = mapped_column(String, nullable=False)
    prediction: Mapped[str] = mapped_column(String, nullable=False)
    date_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False) 