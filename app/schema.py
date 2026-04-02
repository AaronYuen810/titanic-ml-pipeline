from pydantic import BaseModel

class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float

class PredictionOutput(BaseModel):
    survived: int
    probability: float