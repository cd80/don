"""API routes for Don trading framework dashboard.

This module defines the API routes for accessing training metrics
and model information.
"""

from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TrainingMetrics(BaseModel):
    """Training metrics data model."""
    episode: int
    reward: float
    loss: float
    epsilon: float

@router.get("/metrics/latest", response_model=TrainingMetrics)
async def get_latest_metrics() -> TrainingMetrics:
    """Get latest training metrics."""
    # Placeholder: Implement metric storage and retrieval
    return TrainingMetrics(
        episode=0,
        reward=0.0,
        loss=0.0,
        epsilon=1.0,
    )

@router.get("/metrics/history", response_model=List[TrainingMetrics])
async def get_metrics_history() -> List[TrainingMetrics]:
    """Get historical training metrics."""
    # Placeholder: Implement metric history storage and retrieval
    return [
        TrainingMetrics(
            episode=0,
            reward=0.0,
            loss=0.0,
            epsilon=1.0,
        )
    ]

@router.get("/status")
async def get_training_status() -> Dict[str, str]:
    """Get current training status."""
    # Placeholder: Implement training status tracking
    return {"status": "running"}
