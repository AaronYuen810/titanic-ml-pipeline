import pytest
from pydantic import ValidationError
from app.schema import PassengerInput, PredictionOutput


def test_passenger_input_validation():
    """Test that valid data creates a PassengerInput object successfully."""
    valid_data = {
        "Pclass": 1,
        "Sex": "male",
        "Age": 25.0,
    }
    passenger = PassengerInput(**valid_data)
    assert passenger.Pclass == 1
    assert passenger.Sex == "male"
    assert passenger.Age == 25.0


def test_passenger_input_invalid_types():
    """Test that incorrect data types raise a ValidationError."""
    invalid_data = {"Pclass": "One", "Sex": "male", "Age": "twenty five"}
    with pytest.raises(ValidationError):
        PassengerInput(**invalid_data)


def test_passenger_input_missing_fields():
    """Test that missing fields raise a ValidationError."""
    missing_data = {"Pclass": 1, "Sex": "female"}
    with pytest.raises(ValidationError):
        PassengerInput(**missing_data)
