import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from preprocess import TitanicPreprocessor
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib


def main():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    df = pd.read_csv(data_path)

    # Feature selection
    features = ["Pclass", "Sex", "Age"]
    X = df[features]
    y = df["Survived"]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build pipeline using the preprocessor and logistic regression model
    pipeline = Pipeline(
        [
            ("preprocessor", TitanicPreprocessor()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # Model Performance
    print("\n---Validation Model Performance---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.5f}")
    print(f"AUROC: {roc_auc_score(y_val, y_prob):.5f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model_pipeline.joblib")
    joblib.dump(pipeline, model_path)


if __name__ == "__main__":  # entry point
    main()
