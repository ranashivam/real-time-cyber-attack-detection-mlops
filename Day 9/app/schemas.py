
"""
Define request and response schemas for the inference API.
"""

from pydantic import BaseModel, ConfigDict, Field


# Define and validate the five network activity features expected by the model.
class NetworkFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_port: int = Field(ge=0, le=65535)
    bytes_transferred: float = Field(ge=0)
    connection_count: int = Field(ge=0)
    failed_logins: int = Field(ge=0)
    api_requests_per_minute: float = Field(ge=0)


# Define the JSON structure accepted by the prediction endpoint.
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: NetworkFeatures


# Define the JSON structure returned after a successful prediction.
class PredictionResponse(BaseModel):
    prediction: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float


# Define the response returned by the health and readiness endpoints.
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
