# Titanic ML Production Pipeline

This repository demonstrates the end-to-end lifecycle of a Machine Learning project, transitioning from an experimental **Jupyter Notebook** to a production-ready **FastAPI** inference service, fully containerized with **Docker** and deployed to the cloud.

## 🚀 Project Overview

The goal of this project is to showcase the "productionization" of a standard ML model. While the model itself predicts passenger survival on the Titanic, the focus is on the engineering workflow: refactoring code for maintainability, wrapping it in a high-performance API, and ensuring environment parity through containerization.

## 🏗️ The 4-Step Workflow

1.  **Exploration & Training:** Initial data analysis, feature engineering, and model training conducted in a Jupyter Notebook (`notebooks/`).
2.  **Production Refactor:** Transitioning notebook logic into modular Python scripts. This involves creating dedicated modules for preprocessing, model loading, and inference logic within a structured `src/` directory.
3.  **API Serving:** Implementing a **FastAPI** backend to serve model predictions. This provides a robust RESTful interface for real-time inference.
4.  **Deployment & Containerization:** Packaging the application into a **Docker** image to ensure it runs consistently across different environments, followed by deployment to a cloud provider.

## 📁 Project Structure

```text
.
├── notebooks/          # Initial EDA and model prototyping
├── src/                # Production-ready source code
│   ├── main.py         # FastAPI application entry point
│   ├── model.py        # Model loading and prediction logic
│   └── processing.py   # Data transformation and feature engineering
├── models/             # Serialized model files (.pkl or .joblib)
├── Dockerfile          # Instructions for building the container image
├── requirements.txt    # Project dependencies
└── README.md
```

## 🛠️ Tech Stack

* **ML Framework:** Scikit-learn, Pandas, NumPy
* **Backend:** FastAPI, Uvicorn (ASGI Server)
* **DevOps:** Docker
* **Cloud:** [Insert Cloud Provider, e.g., AWS / GCP / Azure]
* **Environment:** Python 3.9+

## 🚦 Getting Started

### Prerequisites
* Docker installed locally
* Python 3.9+ (for local development)

### Local Development
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/titanic-ml-deployment.git](https://github.com/yourusername/titanic-ml-deployment.git)
   cd titanic-ml-deployment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API locally:**
   ```bash
   uvicorn src.main:app --reload
   ```

### Running with Docker
1. **Build the image:**
   ```bash
   docker build -t titanic-prediction-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 titanic-prediction-app
   ```

## 🔌 API Usage

Once the app is running, you can access the interactive API documentation (Swagger UI) at:
`http://localhost:8000/docs`

### Example Request
**POST** `/predict`
```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

## ☁️ Cloud Deployment
This application is configured for deployment via [Insert Deployment Method, e.g., GitHub Actions to AWS App Runner]. The Dockerized nature allows it to scale horizontally to meet demand.
