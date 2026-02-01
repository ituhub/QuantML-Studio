"""
QuantML Studio - REST API v4.0
===============================
FastAPI-based REST API for model predictions and management
"""

import os
import logging
import pickle
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# API MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    use_ensemble: bool = Field(False, description="Use ensemble model if available")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    data: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    use_ensemble: bool = Field(False, description="Use ensemble model if available")


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: Any
    confidence: Optional[float] = None
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[Any]
    model_used: str
    count: int
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Model information response"""
    model_id: str
    model_type: str
    task_type: str
    features: List[str]
    metrics: Dict[str, float]
    created_at: datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: int
    uptime_seconds: float


class TrainRequest(BaseModel):
    """Request model for training"""
    data: List[Dict[str, Any]] = Field(..., description="Training data")
    target_column: str = Field(..., description="Name of target column")
    max_models: int = Field(5, description="Maximum models to train")
    time_limit: int = Field(5, description="Time limit in minutes")


# =============================================================================
# API APPLICATION
# =============================================================================

app = FastAPI(
    title="QuantML Studio API",
    description="Enterprise AI/ML Platform REST API",
    version="4.0.0",
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

# Global state
MODEL_REGISTRY: Dict[str, Any] = {}
API_START_TIME = datetime.now()
MODEL_DIR = Path("models")


# =============================================================================
# AUTHENTICATION
# =============================================================================

API_KEYS = {
    "demo-api-key-12345": "demo",
    "starter-api-key-2024": "starter",
    "pro-api-key-2024": "professional",
    "enterprise-api-key-2024": "enterprise"
}

TIER_LIMITS = {
    "demo": {"predictions_per_day": 100, "batch_size": 10, "models": 1},
    "starter": {"predictions_per_day": 1000, "batch_size": 100, "models": 5},
    "professional": {"predictions_per_day": 10000, "batch_size": 1000, "models": 20},
    "enterprise": {"predictions_per_day": 100000, "batch_size": 10000, "models": 100}
}


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key and return tier"""
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[x_api_key]


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "name": "QuantML Studio API",
        "version": "4.0.0",
        "description": "Enterprise AI/ML Platform",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - API_START_TIME).total_seconds()
    return HealthResponse(
        status="healthy",
        version="4.0.0",
        models_loaded=len(MODEL_REGISTRY),
        uptime_seconds=uptime
    )


@app.get("/models", tags=["Models"])
async def list_models(tier: str = Depends(verify_api_key)):
    """List all available models"""
    models = []
    for model_id, model_data in MODEL_REGISTRY.items():
        models.append({
            "model_id": model_id,
            "model_type": model_data.get("model_type", "unknown"),
            "task_type": model_data.get("task_type", "unknown"),
            "created_at": model_data.get("created_at", datetime.now()).isoformat()
        })
    return {"models": models, "count": len(models), "tier": tier}


@app.get("/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_id: str, tier: str = Depends(verify_api_key)):
    """Get information about a specific model"""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model_data = MODEL_REGISTRY[model_id]
    return ModelInfo(
        model_id=model_id,
        model_type=model_data.get("model_type", "unknown"),
        task_type=model_data.get("task_type", "unknown"),
        features=model_data.get("features", []),
        metrics=model_data.get("metrics", {}),
        created_at=model_data.get("created_at", datetime.now())
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest, tier: str = Depends(verify_api_key)):
    """Make a single prediction"""
    model_id = request.model_id or "default"
    
    if model_id not in MODEL_REGISTRY:
        # Return demo prediction
        prediction = np.random.random() * 100
        return PredictionResponse(
            prediction=float(prediction),
            confidence=0.85,
            model_used="demo_model"
        )
    
    model_data = MODEL_REGISTRY[model_id]
    model = model_data["model"]
    
    try:
        df = pd.DataFrame([request.features])
        
        if request.use_ensemble and "ensemble" in model_data:
            prediction = model_data["ensemble"].predict(df)
            model_used = "ensemble"
        else:
            prediction = model.predict(df)
            model_used = model_data.get("model_type", "unknown")
        
        return PredictionResponse(
            prediction=float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction),
            confidence=model_data.get("confidence", None),
            model_used=model_used
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest, tier: str = Depends(verify_api_key)):
    """Make batch predictions"""
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["demo"])
    
    if len(request.data) > limits["batch_size"]:
        raise HTTPException(
            status_code=403,
            detail=f"Batch size exceeds limit for {tier} tier ({limits['batch_size']})"
        )
    
    model_id = request.model_id or "default"
    
    if model_id not in MODEL_REGISTRY:
        predictions = [float(np.random.random() * 100) for _ in request.data]
        return BatchPredictionResponse(
            predictions=predictions,
            model_used="demo_model",
            count=len(predictions)
        )
    
    model_data = MODEL_REGISTRY[model_id]
    model = model_data["model"]
    
    try:
        df = pd.DataFrame(request.data)
        
        if request.use_ensemble and "ensemble" in model_data:
            predictions = model_data["ensemble"].predict(df).tolist()
            model_used = "ensemble"
        else:
            predictions = model.predict(df).tolist()
            model_used = model_data.get("model_type", "unknown")
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_used=model_used,
            count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train", tags=["Training"])
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks, 
                      tier: str = Depends(verify_api_key)):
    """Start model training (async)"""
    if tier not in ["professional", "enterprise"]:
        raise HTTPException(
            status_code=403,
            detail="Training via API requires Professional or Enterprise tier"
        )
    
    model_id = str(uuid.uuid4())[:8]
    
    # In production, this would trigger async training
    return {
        "status": "training_started",
        "model_id": model_id,
        "message": "Model training has been queued",
        "estimated_time_minutes": request.time_limit
    }


@app.get("/usage", tags=["Usage"])
async def get_usage(tier: str = Depends(verify_api_key)):
    """Get API usage statistics"""
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["demo"])
    
    return {
        "tier": tier,
        "predictions_today": 0,  # Would track in production
        "predictions_this_month": 0,
        "models_deployed": len(MODEL_REGISTRY),
        "limits": limits
    }


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str, tier: str = Depends(verify_api_key)):
    """Delete a model"""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    del MODEL_REGISTRY[model_id]
    return {"status": "deleted", "model_id": model_id}


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def load_model_from_file(filepath: str) -> Dict[str, Any]:
    """Load a model from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def register_model(model_id: str, model_data: Dict[str, Any]):
    """Register a model in the registry"""
    MODEL_REGISTRY[model_id] = {
        **model_data,
        "created_at": datetime.now()
    }
    logger.info(f"Registered model: {model_id}")


def load_all_models():
    """Load all models from the models directory"""
    if MODEL_DIR.exists():
        for model_file in MODEL_DIR.glob("*.pkl"):
            model_id = model_file.stem
            model_data = load_model_from_file(str(model_file))
            if model_data:
                register_model(model_id, model_data)


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting QuantML Studio API...")
    MODEL_DIR.mkdir(exist_ok=True)
    load_all_models()
    logger.info(f"Loaded {len(MODEL_REGISTRY)} models")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down QuantML Studio API...")


# =============================================================================
# RUN SERVER
# =============================================================================

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()
