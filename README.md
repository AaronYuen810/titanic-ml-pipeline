# Titanic ML Production Pipeline

This repository demonstrates the end-to-end lifecycle of a Machine Learning project, transitioning from an experimental **Jupyter Notebook** to a production-ready **FastAPI** inference service, fully containerized with **Docker** and deployed to the cloud.

## 🚀 Project Overview

The goal of this project is to showcase the "productionization" of a standard ML model. While the model itself predicts passenger survival on the Titanic, the focus is on the engineering workflow: refactoring code for maintainability, wrapping it in a high-performance API, and ensuring environment parity through containerization.

## 🏗️ The 4-Step Workflow

1. **Exploration & Training:** Initial data analysis, feature engineering, and model training conducted in a Jupyter Notebook (`notebooks/`).
2. **Production Refactor:** Transitioning notebook logic into modular Python scripts. This involves creating dedicated modules for preprocessing, model loading, and inference logic within a structured `src/` directory.
3. **API Serving:** Implementing a **FastAPI** backend to serve model predictions. This provides a robust RESTful interface for real-time inference.
4. **Deployment & Containerization:** Packaging the application into a **Docker** image to ensure it runs consistently across different environments, followed by deployment to a cloud provider.

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

- **ML Framework:** Scikit-learn, Pandas, NumPy
- **Backend:** FastAPI, Uvicorn (ASGI Server)
- **DevOps:** Docker
- **Cloud:** [Insert Cloud Provider, e.g., AWS / GCP / Azure]
- **Environment:** Python 3.12+

## 🚦 Getting Started

### Prerequisites

- Docker installed locally (BuildKit enabled)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (for local development)

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

## Room for Improvements

| # | Area | What's wrong with current project | Production improvement |
|---|---|---|---|
| 1 | Model build/training lifecycle | The model is trained during Docker image build, coupling training and deployment while making releases slower and less reproducible. | Split training and serving pipelines, store validated model versions in a model registry, and deploy by model version reference. |
| 2 | Data storage and governance | Training data is read from local CSV files with no clear lineage, versioning, or quality gates. | Move datasets to managed storage (data lake/warehouse), enforce schema and quality checks, and version datasets used for each model run. |
| 3 | API input validation | Request fields are type-checked but domain constraints are not enforced (for example: valid class range or allowed values for sex). | Add strict validation rules and enums, return clear validation errors, and version the API contract. |
| 4 | Error handling contract | The prediction endpoint may return an error payload shape that does not match the declared response model. | Use consistent error schemas and HTTP status codes via centralized exception handlers. |
| 5 | Startup/model loading robustness | Model loading failures are not handled as hard startup failures, which can lead to partially healthy service states. | Add fail-fast startup behavior or readiness gating, and separate liveness and readiness probes. |
| 6 | Observability | There is no structured logging, metrics, tracing, or model-serving telemetry for latency and prediction behavior. | Add JSON logs, request IDs, metrics (latency, error rate, throughput), tracing, and model-level monitoring. |
| 7 | Security and API hardening | The API lacks authentication, request throttling, and guardrails for misuse. | Add auth (token/OIDC), rate limiting, payload size limits, strict CORS policy, and run behind an API gateway. |
| 8 | CI/CD quality gates | CI currently focuses on lint/test/build and lacks supply-chain and deployment safety controls. | Add dependency/image vulnerability scanning, SBOM generation, image signing, environment promotion checks, and progressive rollout strategy. |
| 9 | Testing depth | Existing tests are mainly unit-level and do not fully cover integration, performance, and model regression risks. | Add integration tests with real artifacts, contract tests, load tests, and regression tests against baseline model metrics. |
| 10 | Configuration management | Key runtime and path assumptions are hardcoded, reducing portability across environments. | Externalize configuration with environment variables and typed settings, and manage secrets through a secret manager. |
| 11 | Reproducibility and dependency safety | Broad dependency ranges can introduce drift over time and inconsistent behavior across runs. | Enforce locked dependencies in CI/CD, pin runtime base images by digest, and record model/training metadata for reproducibility. |
| 12 | Packaging and architecture hygiene | Runtime path manipulation is used to support model unpickling, which is brittle for long-term maintenance. | Package project modules cleanly, avoid path hacks, and standardize model serialization/loading contracts. |