"""Test suite for Don trading framework security features."""

import os
import tempfile
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

from don.security import (
    KeyRotationManager, RBACManager, RateLimiter,
    SecureConfig, Permission, Role
)
from don.dashboard.app import app

@pytest.fixture
def key_manager():
    """Create temporary key manager for testing."""
    with tempfile.NamedTemporaryFile() as temp_file:
        manager = KeyRotationManager(temp_file.name)
        manager.initialize()
        yield manager

@pytest.fixture
def rbac_manager():
    """Create RBAC manager for testing."""
    return RBACManager()

@pytest.fixture
def rate_limiter():
    """Create rate limiter for testing."""
    return RateLimiter()

@pytest.fixture
def secure_config():
    """Create temporary secure config for testing."""
    with tempfile.NamedTemporaryFile() as temp_file:
        config = SecureConfig(temp_file.name)
        config.load_config()
        yield config

@pytest.fixture
def test_client(key_manager, rbac_manager, rate_limiter):
    """Create test client with security middleware."""
    client = TestClient(app)
    return client

def test_key_rotation(key_manager):
    """Test API key rotation mechanism."""
    # Get initial key
    current_key, _ = key_manager.get_active_keys()
    assert current_key is not None

    # Force rotation
    key_manager._generate_new_key()
    new_key, prev_key = key_manager.get_active_keys()

    # Verify rotation
    assert new_key != current_key
    assert prev_key == current_key
    assert key_manager.validate_key(new_key)
    assert key_manager.validate_key(prev_key)

def test_rbac_permissions(rbac_manager):
    """Test role-based access control."""
    # Assign roles
    user_id = "test-user"
    rbac_manager.assign_role(user_id, Role.TRADER)

    # Check permissions
    permissions = rbac_manager.get_user_permissions(user_id)
    assert Permission.READ_MARKET_DATA in permissions
    assert Permission.READ_TRADES in permissions
    assert Permission.WRITE_TRADES in permissions
    assert Permission.MANAGE_USERS not in permissions

def test_rate_limiting(rate_limiter):
    """Test rate limiting functionality."""
    user_id = "test-user"
    role = Role.TRADER

    # Test within limit
    for _ in range(rate_limiter._rate_limits[role]):
        assert rate_limiter.check_rate_limit(user_id, role)

    # Test exceeding limit
    assert not rate_limiter.check_rate_limit(user_id, role)

    # Test remaining requests
    remaining = rate_limiter.get_remaining_requests(user_id, role)
    assert remaining == 0

def test_secure_config(secure_config):
    """Test secure configuration management."""
    # Test setting and getting values
    secure_config.set("test_key", "test_value")
    assert secure_config.get("test_key") == "test_value"

    # Test encryption rotation
    secure_config._rotate_config()
    assert secure_config.get("test_key") == "test_value"

def test_protected_endpoint(test_client, key_manager, rbac_manager):
    """Test protected API endpoint."""
    # Get valid API key
    api_key, _ = key_manager.get_active_keys()

    # Test without API key
    response = test_client.get("/api/market-data")
    assert response.status_code == 401

    # Test with invalid API key
    headers = {"X-API-Key": "invalid-key"}
    response = test_client.get("/api/market-data", headers=headers)
    assert response.status_code == 401

    # Test with valid API key but no role
    headers = {"X-API-Key": api_key}
    response = test_client.get("/api/market-data", headers=headers)
    assert response.status_code == 403

    # Test with valid API key and role
    user_id = "test-user"
    rbac_manager.assign_role(user_id, Role.TRADER)
    response = test_client.get("/api/market-data", headers=headers)
    assert response.status_code == 200
