
# Day 9 - FastAPI Model Inference Service

## Objective

Build a FastAPI application that exposes the approved Autoencoder
model through HTTP endpoints.

## Project Structure

```text
Day 9/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── artifacts/
│   └── model/
│       └── approved_model.joblib
├── requirements.txt
└── README.md
```

## Model

The approved Autoencoder model is reused from Day 8.

The model is loaded once when the FastAPI application starts.

## Start the API

Run from the Day 9 project directory:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API Documentation

Open:

http://127.0.0.1:8000/docs

## Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Check application health |
| GET | /ready | Check model readiness |
| POST | /predict | Predict network activity anomaly |

## Example Prediction Request

```json
{
  "features": {
    "destination_port": 443,
    "bytes_transferred": 1200,
    "connection_count": 5,
    "failed_logins": 0,
    "api_requests_per_minute": 20
  }
}
```

## Example Prediction Response

```json
{
  "prediction": "NORMAL",
  "is_anomaly": false,
  "anomaly_score": 0.9054298670852405,
  "threshold": 1.316508753136059
}
```

The example response shows the previously observed Day 8 normal
sample. Actual results depend on the loaded model artifact.

## Stop the API

Press Ctrl+C in the terminal running Uvicorn.
