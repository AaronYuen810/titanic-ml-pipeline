import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from typing import List

# Ensure the unpickler can locate `preprocess.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from app.schema import PassengerInput, PredictionOutput

app = FastAPI(title="Titanic Predictor", version="1.0.0")

# Load model pipeline on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_pipeline.joblib')
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Warning: Model not found at {MODEL_PATH}.")

@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=List[PredictionOutput])
def predict(passengers: List[PassengerInput]):
    """
    Predict survival given passenger features.
    """
    if pipeline is None:
        return {"error": "Model uninitialized or not found."}
        
    df = pd.DataFrame([p.model_dump() for p in passengers])
    
    # Generate predictions and probabilities
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]
    
    # Format and return outputs
    results = [
        PredictionOutput(survived=int(pred), probability=float(prob))
        for pred, prob in zip(predictions, probabilities)
    ]
    
    return results
