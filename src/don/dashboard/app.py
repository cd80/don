"""FastAPI application for Don trading framework dashboard.

This module provides the FastAPI application instance with CORS middleware
for serving training metrics and visualization.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

# Create FastAPI instance
app = FastAPI(title="Don Trading Dashboard")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routes
app.include_router(router, prefix="/api")
