"""
Prometheus metrics for monitoring API and model performance.
"""

import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import Response
import time

logger = logging.getLogger(__name__)

# Define metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint']
)

analysis_mood_distribution = Counter(
    'analysis_mood_distribution',
    'Distribution of detected moods',
    ['mood_category']
)

analysis_motivation_gauge = Gauge(
    'analysis_motivation_level',
    'Current motivation level from latest analysis'
)

prediction_errors_total = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)


class MetricsMiddleware:
    """Middleware for tracking API metrics."""
    
    def __init__(self):
        pass
    
    def track_request(self, endpoint: str, method: str, status: int, duration: float):
        """Track an API request."""
        api_requests_total.labels(endpoint=endpoint, method=method, status=status).inc()
        api_request_duration_seconds.labels(endpoint=endpoint).observe(duration)
    
    def track_analysis(self, mood: str, motivation_level: float):
        """Track mood analysis results."""
        analysis_mood_distribution.labels(mood_category=mood).inc()
        analysis_motivation_gauge.set(motivation_level)
    
    def track_error(self):
        """Track a prediction error."""
        prediction_errors_total.inc()


def get_metrics() -> Response:
    """Get Prometheus metrics."""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")
