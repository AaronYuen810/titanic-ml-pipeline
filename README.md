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
├── app/                # FastAPI application (e.g. main.py, schema)
├── data/               # Training CSVs
├── src/                # Training and preprocessing (train.py, preprocess.py)
├── models/             # Serialized model (model_pipeline.joblib); produced by training or Docker build
├── Dockerfile          # Multi-stage image: train, then API
├── docker-compose.yml  # Local run: build + port 8000
├── pyproject.toml      # Dependencies (locked with uv.lock)
└── README.md
```

## 🛠️ Tech Stack

* **ML Framework:** Scikit-learn, Pandas, NumPy
* **Backend:** FastAPI, Uvicorn (ASGI Server)
* **DevOps:** Docker
* **Cloud:** [Insert Cloud Provider, e.g., AWS / GCP / Azure]
* **Environment:** Python 3.12+

## 🚦 Getting Started

### Prerequisites
* Docker installed locally (BuildKit enabled)
* Python 3.12+ and [uv](https://docs.astral.sh/uv/) (for local development)

### Local Development
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/titanic-ml-deployment.git](https://github.com/yourusername/titanic-ml-deployment.git)
   cd titanic-ml-deployment
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the API locally** (from the repo root, after training has produced `models/model_pipeline.joblib`):
   ```bash
   uv run uvicorn app.main:app --reload
   ```

### Running with Docker
The image is **multi-stage**: it trains the model from `data/train.csv` in the build, then runs the FastAPI app with the trained `models/model_pipeline.joblib`.

1. **Build the image:**
   ```bash
   docker build -t titanic-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 titanic-api
   ```

3. **Or use Compose:**
   ```bash
   docker compose up --build
   ```

## 🔌 API Usage

Once the app is running, you can access the interactive API documentation (Swagger UI) at:
`http://localhost:8000/docs`

### Example Request
**POST** `/predict` (body is a JSON array of passengers)

```json
[
  {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0
  }
]
```

## ☁️ Cloud Deployment
This application is configured for deployment via [Insert Deployment Method, e.g., GitHub Actions to AWS App Runner]. The Dockerized nature allows it to scale horizontally to meet demand.
