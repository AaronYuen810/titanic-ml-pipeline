import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import app.main as main
from app.main import app
from app.schema import PredictionOutput
from src.preprocess import TitanicPreprocessor


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    # Inject a tiny in-memory fitted pipeline so tests don't depend on
    # models/model_pipeline.joblib
    X_train = pd.DataFrame(
        [
            {"Pclass": 1, "Sex": "male", "Age": 10.0},
            {"Pclass": 3, "Sex": "female", "Age": 30.0},
            {"Pclass": 2, "Sex": "male", "Age": 35.0},
            {"Pclass": 3, "Sex": "female", "Age": 5.0},
        ]
    )
    y_train = [0, 1, 1, 0]

    pipeline = Pipeline(
        [
            ("preprocessor", TitanicPreprocessor()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    monkeypatch.setattr(main, "pipeline", pipeline, raising=True)

    with TestClient(app) as c:
        yield c


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_endpoint(client: TestClient):
    payload = [
        {
            "Pclass": 1,
            "Sex": "male",
            "Age": 25.0,
        },
        {
            "Pclass": 2,
            "Sex": "female",
            "Age": 30.0,
        },
    ]
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert len(data) == len(payload)

    for d in data:
        prediction_output = PredictionOutput(**d)
        assert prediction_output.survived in (0, 1)
        assert 0.0 <= prediction_output.probability <= 1.0


def test_predict_empty_input(client):
    # Testing an empty list or bad data
    response = client.post("/predict", json=[{}])
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_missing_data(client):
    # Testing an empty list or bad data

    missing_data = [{"Pclass": 1}]
    response = client.post("/predict", json=missing_data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_wrong_types(client: TestClient):
    wrong_types = [
        {"Pclass": "One", "Sex": "male", "Age": "xx"},
    ]
    response = client.post("/predict", json=wrong_types)
    assert response.status_code == 422
