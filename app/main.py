import os
import sys
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI

# Add the project root to the Python path so that the unpickler can find `preprocess.py`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from app.schema import PassengerInput, PredictionOutput

# Load the model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model_pipeline.joblib"
)
pipeline = None


def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: Model file not found. Please check if model path.")


async def lifespan(app: FastAPI):
    # load model on startup
    load_model()
    yield

    # clear model on shutdown
    global pipeline
    pipeline = None


# Create FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
    description=(
        "Predict whether a passenger survived the Titanic disaster based on their "
        "characteristics"
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Check the health of the API and the model.
    """
    return {"status": "healthy", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=List[PredictionOutput])
def predict(passengers: List[PassengerInput]):
    """Predict the survival of a list of passengers"""

    if pipeline is None:
        return {"error": "Model is not loaded."}

    df = pd.DataFrame([p.model_dump() for p in passengers])

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    results = [
        PredictionOutput(survived=pred, probability=prob)
        for pred, prob in zip(predictions, probabilities)
    ]

    return results
