"""
FastAPI Application for CIFAR-10 Model Inference

This module provides a REST API for CIFAR-10 image classification with monitoring and validation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
import yaml

# Import model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cifar_cnn import CIFAR10CNN

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load API configuration from YAML file."""
    config_path = "configs/api.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            "server": {"host": "0.0.0.0", "port": 8000},
            "model": {"path": "models/best_model.pth", "device": "cpu"},
            "monitoring": {"enable_metrics": True}
        }

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="CIFAR-10 Classification API",
    description="A REST API for CIFAR-10 image classification using CNN",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("security", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
if config.get("monitoring", {}).get("enable_metrics", True):
    REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
    REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
    PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
    PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')

# Global model instance
model = None
model_info = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    image: List[List[List[float]]] = Field(..., description="32x32x3 RGB image as 3D array")
    confidence_threshold: Optional[float] = Field(0.5, description="Minimum confidence threshold")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0-9)")
    prediction_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score")
    probabilities: List[float] = Field(..., description="Class probabilities")
    class_names: List[str] = Field(..., description="Class names")
    processing_time: float = Field(..., description="Processing time in seconds")

class ModelInfo(BaseModel):
    model_name: str
    architecture: str
    input_size: List[int]
    num_classes: int
    total_parameters: int
    device: str
    class_names: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

def load_model():
    """Load the trained model."""
    global model, model_info
    
    try:
        model_path = config["model"]["path"]
        device = config["model"]["device"]
        
        if not os.path.exists(model_path):
            logger.error("Model file not found", path=model_path)
            return False
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model instance
        model_config = checkpoint.get('config', {}).get('model', {})
        model = CIFAR10CNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Get model info
        model_info = checkpoint.get('model_info', {})
        model_info['device'] = device
        model_info['class_names'] = CIFAR10_CLASSES
        
        logger.info("Model loaded successfully", 
                   path=model_path, 
                   device=device,
                   model_info=model_info)
        return True
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        return False

def preprocess_image(image_data: List[List[List[float]]]) -> torch.Tensor:
    """Preprocess image data for model inference."""
    try:
        # Convert to numpy array
        image_array = np.array(image_data, dtype=np.float32)
        
        # Ensure correct shape (32x32x3)
        if image_array.shape != (32, 32, 3):
            raise ValueError(f"Expected image shape (32, 32, 3), got {image_array.shape}")
        
        # Normalize to [0, 1] if not already
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Convert to tensor and rearrange dimensions (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Apply CIFAR-10 normalization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        logger.error("Failed to preprocess image", error=str(e))
        raise ValueError(f"Invalid image data: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting CIFAR-10 Classification API")
    
    # Load model
    if not load_model():
        logger.warning("Model not loaded - some endpoints may not work")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CIFAR-10 Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "dataset": "CIFAR-10",
        "classes": "10 classes: " + ", ".join(CIFAR10_CLASSES)
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import datetime
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(**model_info)

@app.get("/classes")
async def get_classes():
    """Get CIFAR-10 class names and indices."""
    return {
        "classes": {i: name for i, name in enumerate(CIFAR10_CLASSES)},
        "num_classes": len(CIFAR10_CLASSES)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction on a CIFAR-10 image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(request.image)
        input_tensor = input_tensor.to(model.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # Convert to Python types
        prediction = prediction.item()
        confidence = confidence.item()
        probabilities = probabilities.squeeze().cpu().numpy().tolist()
        
        processing_time = time.time() - start_time
        
        # Update metrics
        if config.get("monitoring", {}).get("enable_metrics", True):
            PREDICTION_COUNT.inc()
            PREDICTION_LATENCY.observe(processing_time)
        
        # Check confidence threshold
        if confidence < request.confidence_threshold:
            logger.warning("Low confidence prediction", 
                         prediction=prediction, 
                         confidence=confidence,
                         threshold=request.confidence_threshold)
        
        logger.info("Prediction made", 
                   prediction=prediction, 
                   prediction_name=CIFAR10_CLASSES[prediction],
                   confidence=confidence,
                   processing_time=processing_time)
        
        return PredictionResponse(
            prediction=prediction,
            prediction_name=CIFAR10_CLASSES[prediction],
            confidence=confidence,
            probabilities=probabilities,
            class_names=CIFAR10_CLASSES,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions on multiple CIFAR-10 images."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    results = []
    
    try:
        for i, request in enumerate(requests):
            # Preprocess image
            input_tensor = preprocess_image(request.image)
            input_tensor = input_tensor.to(model.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            # Convert to Python types
            prediction = prediction.item()
            confidence = confidence.item()
            probabilities = probabilities.squeeze().cpu().numpy().tolist()
            
            results.append({
                "index": i,
                "prediction": prediction,
                "prediction_name": CIFAR10_CLASSES[prediction],
                "confidence": confidence,
                "probabilities": probabilities
            })
        
        processing_time = time.time() - start_time
        
        logger.info("Batch prediction completed", 
                   batch_size=len(requests),
                   processing_time=processing_time)
        
        return {
            "predictions": results,
            "batch_size": len(requests),
            "processing_time": processing_time,
            "class_names": CIFAR10_CLASSES
        }
        
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not config.get("monitoring", {}).get("enable_metrics", True):
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    return JSONResponse(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Update metrics
    if config.get("monitoring", {}).get("enable_metrics", True):
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
        REQUEST_LATENCY.observe(process_time)
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", 
                error=str(exc), 
                path=request.url.path,
                method=request.method)
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"].get("reload", False)
    )
