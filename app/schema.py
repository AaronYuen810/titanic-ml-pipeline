from pydantic import BaseModel
from typing import Optional

class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: Optional[float] = None

class PredictionOutput(BaseModel):
    survived: int
    probability: float
