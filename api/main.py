"""
Traffic Violation Prediction API

Provides REST endpoints for:
- Predicting traffic violation types
- Model information and health checks

Author: VaishnaBala
"""

import os
import sys
import pickle
import numpy as np # type: ignore
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel, Field
from typing import Optional

# ============ PATH SETUP ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

sys.path.insert(0, PROJECT_ROOT)
# ====================================


# Feature columns (must match training)
FEATURE_COLS = [
    "coords_long_scaled", "coords_lat_scaled",
    "hour", "day_of_week",
    "is_weekend", "is_peak_hour", "is_night",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "junctionName_encoded", "vehicleType_encoded",
    "junction_risk_score", "vehicle_risk_score"
]

# Junction and vehicle mappings
JUNCTIONS = {
    "JPnagar-8thmain-9thcross": {"coords": [77.595183, 12.910289], "risk": 0.7},
    "Koramangala-5thblock": {"coords": [77.612320, 12.935620], "risk": 0.8},
    "Whitefield-main-road": {"coords": [77.750277, 12.969720], "risk": 0.5},
    "MG-Road-brigade": {"coords": [77.606430, 12.975040], "risk": 0.9},
    "Indiranagar-100ft-road": {"coords": [77.640877, 12.978012], "risk": 0.75},
    "Electronic-city-toll": {"coords": [77.670067, 12.839547], "risk": 0.4},
    "Hebbal-flyover": {"coords": [77.591508, 13.035542], "risk": 0.85},
    "Silk-board-junction": {"coords": [77.622627, 12.917153], "risk": 0.95}
}

VEHICLES = {
    "Motorbike": 0.8,
    "Car": 0.5,
    "Bus": 0.3,
    "Truck": 0.4,
    "Auto": 0.6
}

ALERT_TYPES = ["RED_LIGHT_VIOLATION", "SPEED_VIOLATION", "WRONG_WAY"]


# ============ LOAD MODEL & ENCODERS ============
def load_artifacts():
    """Load model and encoders."""
    
    model_path = os.path.join(MODELS_DIR, "best_baseline_model.pkl")
    encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders not found: {encoders_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    
    return model, encoders


# Load on startup
try:
    MODEL, ENCODERS = load_artifacts()
    MODEL_LOADED = True
    print("✅ Model and encoders loaded successfully")
except Exception as e:
    MODEL, ENCODERS = None, None
    MODEL_LOADED = False
    print(f"❌ Failed to load model: {e}")


# ============ FASTAPI APP ============
app = FastAPI(
    title="Traffic Violation Prediction API",
    description="""
    ## 🚦 Traffic Violation Prediction System
    
    An ML-powered API for predicting traffic violation types based on 
    location, time, and vehicle information.
    
    ### Features:
    - **Predict** violation type (Red Light, Speed, Wrong Way)
    - **Risk assessment** based on junction and vehicle type
    - **Explainable** predictions with confidence scores
    
    ### Research Components:
    - XGBoost classifier with cross-validation
    - SHAP & LIME explainability
    - Anomaly detection for unusual patterns
    
    **Author:** VaishnaBala
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ REQUEST/RESPONSE MODELS ============
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    junction_name: str = Field(
        ...,
        description="Name of the junction",
        example="Silk-board-junction"
    ) # type: ignore
    vehicle_type: str = Field(
        ...,
        description="Type of vehicle",
        example="Motorbike"
    ) # type: ignore
    hour: int = Field(
        ...,
        ge=0, le=23,
        description="Hour of day (0-23)",
        example=18
    ) # type: ignore
    day_of_week: int = Field(
        ...,
        ge=0, le=6,
        description="Day of week (0=Monday, 6=Sunday)",
        example=4
    ) # type: ignore
    
    class Config:
        schema_extra = {
            "example": {
                "junction_name": "Silk-board-junction",
                "vehicle_type": "Motorbike",
                "hour": 18,
                "day_of_week": 4
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    predicted_violation: str
    confidence: float
    risk_level: str
    junction_risk: float
    vehicle_risk: float
    is_peak_hour: bool
    probabilities: dict
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_type: str
    n_features: int
    feature_names: list
    supported_junctions: list
    supported_vehicles: list
    alert_types: list


# ============ HELPER FUNCTIONS ============
def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """
    Convert API request to model features.
    
    Args:
        request: PredictionRequest object
    
    Returns:
        DataFrame with features matching model input
    """
    # Validate inputs
    if request.junction_name not in JUNCTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid junction. Choose from: {list(JUNCTIONS.keys())}"
        )
    
    if request.vehicle_type not in VEHICLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vehicle type. Choose from: {list(VEHICLES.keys())}"
        )
    
    # Get junction info
    junction_info = JUNCTIONS[request.junction_name]
    coords = junction_info["coords"]
    junction_risk = junction_info["risk"]
    vehicle_risk = VEHICLES[request.vehicle_type]
    
    # Encode categorical variables
    junction_encoded = ENCODERS["junctionName"].transform([request.junction_name])[0] # type: ignore
    vehicle_encoded = ENCODERS["vehicleType"].transform([request.vehicle_type])[0] # type: ignore
    
    # Scale coordinates
    coord_scaler = ENCODERS["coord_scaler"] # type: ignore
    coords_scaled = coord_scaler.transform([[coords[0], coords[1]]])[0]
    
    # Time features
    hour = request.hour
    day_of_week = request.day_of_week
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_peak_hour = 1 if hour in [8, 9, 10, 17, 18, 19] else 0
    is_night = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Build feature dict
    features = {
        "coords_long_scaled": coords_scaled[0],
        "coords_lat_scaled": coords_scaled[1],
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "is_night": is_night,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "junctionName_encoded": junction_encoded,
        "vehicleType_encoded": vehicle_encoded,
        "junction_risk_score": junction_risk,
        "vehicle_risk_score": vehicle_risk
    }
    
    return pd.DataFrame([features])[FEATURE_COLS] # type: ignore


def get_risk_level(confidence: float, junction_risk: float, vehicle_risk: float) -> str:
    """Determine overall risk level."""
    
    combined_risk = (confidence * 0.4) + (junction_risk * 0.3) + (vehicle_risk * 0.3)
    
    if combined_risk > 0.7:
        return "HIGH"
    elif combined_risk > 0.5:
        return "MEDIUM"
    else:
        return "LOW"


# ============ API ENDPOINTS ============
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🚦 Traffic Violation Prediction API",
        "version": "1.0.0",
        "author": "VaishnaBala",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get model information and supported values."""
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=type(MODEL).__name__,
        n_features=len(FEATURE_COLS),
        feature_names=FEATURE_COLS,
        supported_junctions=list(JUNCTIONS.keys()),
        supported_vehicles=list(VEHICLES.keys()),
        alert_types=ALERT_TYPES
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict traffic violation type.
    
    Takes junction, vehicle type, and time information to predict
    the most likely violation type with confidence score.
    """
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features(request)
        
        # Get prediction
        prediction = MODEL.predict(features)[0] # type: ignore
        probabilities = MODEL.predict_proba(features)[0] # type: ignore
        
        # Decode prediction
        predicted_violation = ENCODERS["alertType"].inverse_transform([prediction])[0] # type: ignore
        confidence = float(probabilities.max())
        
        # Get risk info
        junction_risk = JUNCTIONS[request.junction_name]["risk"]
        vehicle_risk = VEHICLES[request.vehicle_type]
        risk_level = get_risk_level(confidence, junction_risk, vehicle_risk)
        
        # Build probability dict
        prob_dict = {
            alert: float(prob) 
            for alert, prob in zip(ALERT_TYPES, probabilities)
        }
        
        return PredictionResponse(
            predicted_violation=predicted_violation,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            junction_risk=junction_risk,
            vehicle_risk=vehicle_risk,
            is_peak_hour=request.hour in [8, 9, 10, 17, 18, 19],
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/junctions", tags=["Info"])
async def get_junctions():
    """Get list of supported junctions with risk scores."""
    return {
        name: {"risk_score": info["risk"], "coordinates": info["coords"]}
        for name, info in JUNCTIONS.items()
    }


@app.get("/vehicles", tags=["Info"])
async def get_vehicles():
    """Get list of supported vehicle types with risk scores."""
    return {
        name: {"risk_score": risk}
        for name, risk in VEHICLES.items()
    }


# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn # type: ignore
    
    print("\n" + "=" * 50)
    print("🚦 TRAFFIC VIOLATION PREDICTION API")
    print("=" * 50)
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"📁 Models Dir: {MODELS_DIR}")
    print(f"✅ Model Loaded: {MODEL_LOADED}")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )