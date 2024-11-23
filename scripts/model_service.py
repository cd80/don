#!/usr/bin/env python3
"""
Model Service for Bitcoin Trading RL.
Provides a FastAPI service for model inference with GPU support.
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, start_http_server

from src.models.base_model import BaseModel as TradingModel
from src.utils.helpers import setup_logging

# Configure logging
logger = setup_logging(__name__)

# Initialize FastAPI app
app = FastAPI(title="Bitcoin Trading RL Model Service")

# Metrics
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction'
)
PREDICTION_REQUESTS = Counter(
    'model_prediction_requests_total',
    'Total number of prediction requests'
)
ERROR_COUNTER = Counter(
    'model_errors_total',
    'Total number of model errors',
    ['error_type']
)

class PredictionRequest(BaseModel):
    """Prediction request model."""
    features: List[List[float]]
    additional_context: Optional[Dict] = None

class PredictionResponse(BaseModel):
    """Prediction response model."""
    predictions: List[Dict[str, float]]
    metadata: Dict

class ModelService:
    """Model service for handling predictions."""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize model service."""
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_model(self) -> TradingModel:
        """Load the trading model."""
        try:
            model = TradingModel()
            checkpoint_path = os.getenv('MODEL_CHECKPOINT_PATH', 'checkpoints/latest.pt')
            
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found at {checkpoint_path}, using untrained model")
            
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            ERROR_COUNTER.labels(error_type='model_loading').inc()
            raise
    
    @PREDICTION_LATENCY.time()
    def predict(self, features: List[List[float]], context: Optional[Dict] = None) -> List[Dict[str, float]]:
        """Make predictions using the model."""
        try:
            PREDICTION_REQUESTS.inc()
            
            # Convert features to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model.predict(features_tensor)
            
            # Convert predictions to list of dictionaries
            result = []
            for pred in predictions:
                result.append({
                    'action': float(pred['action']),
                    'confidence': float(pred['confidence']),
                    'position_size': float(pred['position_size'])
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            ERROR_COUNTER.labels(error_type='prediction').inc()
            raise

# Initialize model service
model_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model service on startup."""
    global model_service
    try:
        # Start Prometheus metrics server
        start_http_server(8002)
        
        # Initialize model service
        model_service = ModelService(use_gpu=args.gpu)
        logger.info("Model service initialized successfully")
    
    except Exception as e:
        logger.error(f"Error initializing model service: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    return {"status": "healthy", "device": str(model_service.device)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions endpoint."""
    try:
        predictions = model_service.predict(
            request.features,
            request.additional_context
        )
        
        return PredictionResponse(
            predictions=predictions,
            metadata={
                "model_version": os.getenv("MODEL_VERSION", "unknown"),
                "device": str(model_service.device)
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model Service')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging level
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.getLogger().setLevel(log_level)
    
    # Start server
    uvicorn.run(
        "model_service:app",
        host=args.host,
        port=args.port,
        log_level=log_level.lower(),
        workers=1  # Use single worker for GPU model
    )
