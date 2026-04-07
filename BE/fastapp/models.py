from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Boolean, ForeignKey
from datetime import datetime

class Base(DeclarativeBase):
    pass

class PredDB(Base):
    __tablename__ = "predictions"

    pred_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    image_name: Mapped[str] = mapped_column(String, nullable=False)
    prediction: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    date_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    user_name: Mapped[str] = mapped_column(String, nullable=False)
  

class UserDB(Base):
    __tablename__ = "users"

    user_id : Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email : Mapped[str] = mapped_column(String, nullable=False)
    hashed_password : Mapped[str] = mapped_column(String, nullable=False)
    is_active : Mapped[bool] = mapped_column(Boolean, nullable=False)