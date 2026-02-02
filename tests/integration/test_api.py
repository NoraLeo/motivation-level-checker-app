"""Integration tests for the API."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_analyze_positive_entry():
    """Test analysis of positive journal entry."""
    payload = {
        "text": "Today was fantastic! I'm so motivated and accomplished all my goals."
    }
    
    response = client.post("/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "mood" in data
    assert "motivation_level" in data
    assert "motivation_category" in data
    assert data["mood"] in ["positive", "very_positive"]
    assert data["motivation_category"] == "high"


def test_analyze_negative_entry():
    """Test analysis of negative journal entry."""
    payload = {
        "text": "Feeling really unmotivated and lazy. Can't get anything done."
    }
    
    response = client.post("/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["mood"] in ["negative", "very_negative"]
    assert data["motivation_category"] == "low"


def test_analyze_with_preprocessing():
    """Test analysis with preprocessing enabled."""
    payload = {
        "text": "Check out https://example.com - I'm feeling GREAT!",
        "preprocess": True
    }
    
    response = client.post("/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "mood" in data


def test_analyze_empty_text():
    """Test analysis with empty text."""
    payload = {
        "text": ""
    }
    
    response = client.post("/analyze", json=payload)
    
    # Should fail validation
    assert response.status_code == 422


def test_get_stats():
    """Test stats endpoint."""
    response = client.get("/stats")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "mood_categories" in data
    assert "motivation_levels" in data
    assert "available_features" in data
