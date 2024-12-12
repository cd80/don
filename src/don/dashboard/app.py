"""FastAPI application for Don trading framework dashboard.

This module provides the FastAPI application instance with security middleware
for authentication, rate limiting, and RBAC.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from ..security import (
    KeyRotationManager, RBACManager, RateLimiter, SecureConfig,
    Permission
)
from ..security.middleware import SecurityMiddleware, check_permission
from .routes import router

# Initialize security components
key_manager = KeyRotationManager("/etc/don/keys.enc")
rbac_manager = RBACManager()
rate_limiter = RateLimiter()
config = SecureConfig("/etc/don/config.enc")

# Create FastAPI instance
app = FastAPI(
    title="Don Trading Dashboard",
    description="API for Don trading framework with security features",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add security middleware
app.add_middleware(
    SecurityMiddleware,
    key_manager=key_manager,
    rbac_manager=rbac_manager,
    rate_limiter=rate_limiter
)

# Initialize security components
@app.on_event("startup")
async def startup_event():
    key_manager.initialize()
    config.load_config()

# Include routes with security
app.include_router(
    router,
    prefix="/api",
    dependencies=[Depends(check_permission(Permission.READ_MARKET_DATA))]
)
