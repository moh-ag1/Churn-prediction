# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using **XGBoost**.

### Features

* Data validation & preprocessing
* Feature engineering
* XGBoost classification
* Optuna hyperparameter tuning
* MLflow experiment tracking
* FastAPI prediction API
* Gradio web interface
* Docker deployment

### Run locally

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
uvicorn main:app --reload
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`
UI: `http://localhost:8000/ui`

### Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

### Structure

```text
src/          # ML code
scripts/      # Training & testing
artifacts/    # Trained model
main.py       # FastAPI + Gradio
Dockerfile    # Docker configuration
```
