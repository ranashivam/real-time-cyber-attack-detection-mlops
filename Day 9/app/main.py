
"""
Main FastAPI application for cyber-attack anomaly inference.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_loader import load_model_bundle, predict_anomaly
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger("cyber_attack_inference")

model_bundle = None


# Load the approved model when FastAPI starts and release it at shutdown.
@asynccontextmanager
async def lifespan(application: FastAPI):
    global model_bundle

    logger.info("Starting cyber-attack inference service.")

    try:
        model_bundle = load_model_bundle()
        logger.info("Approved Autoencoder model loaded successfully.")

        yield

    except Exception:
        logger.exception("Failed to start the inference service.")
        raise

    finally:
        model_bundle = None
        logger.info("Inference service has stopped.")


app = FastAPI(
    title="Cyber-Attack Detection Inference API",
    description=(
        "A FastAPI service that uses an approved Autoencoder "
        "to identify potentially anomalous network activity."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# Return the service health and indicate whether the model is loaded.
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_bundle is not None
    )


# Confirm that the application and model are ready for predictions.
@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["Health"]
)
async def readiness_check():
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service is not ready. Model is not loaded."
        )

    return HealthResponse(
        status="ready",
        model_loaded=True
    )


# Accept validated network features and return the model prediction.
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
async def predict(request: PredictionRequest):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available."
        )

    try:
        feature_values = request.features.model_dump()

        result = predict_anomaly(
            model_bundle,
            feature_values
        )

        logger.info(
            "Prediction completed: %s",
            result["prediction"]
        )

        return PredictionResponse(**result)

    except ValueError as error:
        logger.warning("Invalid prediction input: %s", error)

        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception("Unexpected prediction error.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction."
        ) from error
