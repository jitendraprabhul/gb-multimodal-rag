"""
Monitoring and metrics collection for the API.

Provides:
- Prometheus-style metrics
- Performance monitoring
- Request tracking
- Health metrics
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import Request
from pydantic import BaseModel


class MetricsCollector:
    """
    Collects and stores application metrics.

    In production, this should integrate with Prometheus, Datadog, or similar.
    For now, we use in-memory storage.
    """

    def __init__(self):
        """Initialize metrics collector."""
        # Request counters
        self.request_count: Dict[str, int] = defaultdict(int)
        self.request_errors: Dict[str, int] = defaultdict(int)

        # Latency tracking
        self.request_latencies: Dict[str, List[float]] = defaultdict(list)
        self.max_latency_samples = 1000  # Keep last N samples

        # Status code tracking
        self.status_codes: Dict[int, int] = defaultdict(int)

        # Component metrics
        self.query_count = 0
        self.ingestion_count = 0
        self.vector_search_count = 0
        self.graph_traversal_count = 0

        # Component latencies
        self.query_latencies: List[float] = []
        self.ingestion_latencies: List[float] = []
        self.vector_search_latencies: List[float] = []
        self.graph_traversal_latencies: List[float] = []

        # Error tracking
        self.errors_by_type: Dict[str, int] = defaultdict(int)
        self.recent_errors: List[Dict] = []
        self.max_recent_errors = 100

        # Start time for uptime calculation
        self.start_time = datetime.utcnow()

    def track_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        error: str = None,
    ) -> None:
        """
        Track an API request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            error: Error message if request failed
        """
        endpoint = f"{method} {path}"

        # Increment request counter
        self.request_count[endpoint] += 1

        # Track status code
        self.status_codes[status_code] += 1

        # Track latency
        latencies = self.request_latencies[endpoint]
        latencies.append(duration_ms)

        # Keep only recent samples
        if len(latencies) > self.max_latency_samples:
            latencies.pop(0)

        # Track errors
        if error or status_code >= 400:
            self.request_errors[endpoint] += 1

            if error:
                self.errors_by_type[error] += 1
                self.recent_errors.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "endpoint": endpoint,
                        "error": error,
                        "status_code": status_code,
                    }
                )

                # Keep only recent errors
                if len(self.recent_errors) > self.max_recent_errors:
                    self.recent_errors.pop(0)

    def track_component(
        self,
        component: str,
        duration_ms: float,
    ) -> None:
        """
        Track component operation.

        Args:
            component: Component name (query, ingestion, vector_search, graph_traversal)
            duration_ms: Operation duration in milliseconds
        """
        if component == "query":
            self.query_count += 1
            self.query_latencies.append(duration_ms)
            if len(self.query_latencies) > self.max_latency_samples:
                self.query_latencies.pop(0)

        elif component == "ingestion":
            self.ingestion_count += 1
            self.ingestion_latencies.append(duration_ms)
            if len(self.ingestion_latencies) > self.max_latency_samples:
                self.ingestion_latencies.pop(0)

        elif component == "vector_search":
            self.vector_search_count += 1
            self.vector_search_latencies.append(duration_ms)
            if len(self.vector_search_latencies) > self.max_latency_samples:
                self.vector_search_latencies.pop(0)

        elif component == "graph_traversal":
            self.graph_traversal_count += 1
            self.graph_traversal_latencies.append(duration_ms)
            if len(self.graph_traversal_latencies) > self.max_latency_samples:
                self.graph_traversal_latencies.pop(0)

    def get_metrics(self) -> dict:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary of all metrics
        """
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()

        # Calculate percentiles
        def percentiles(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}

            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "p50": sorted_values[int(n * 0.5)] if n > 0 else 0,
                "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
                "mean": sum(sorted_values) / n if n > 0 else 0,
            }

        return {
            "uptime_seconds": uptime_seconds,
            "requests": {
                "total": sum(self.request_count.values()),
                "by_endpoint": dict(self.request_count),
                "errors": {
                    "total": sum(self.request_errors.values()),
                    "by_endpoint": dict(self.request_errors),
                },
                "status_codes": dict(self.status_codes),
                "latency": {
                    endpoint: percentiles(latencies)
                    for endpoint, latencies in self.request_latencies.items()
                },
            },
            "components": {
                "query": {
                    "count": self.query_count,
                    "latency": percentiles(self.query_latencies),
                },
                "ingestion": {
                    "count": self.ingestion_count,
                    "latency": percentiles(self.ingestion_latencies),
                },
                "vector_search": {
                    "count": self.vector_search_count,
                    "latency": percentiles(self.vector_search_latencies),
                },
                "graph_traversal": {
                    "count": self.graph_traversal_count,
                    "latency": percentiles(self.graph_traversal_latencies),
                },
            },
            "errors": {
                "by_type": dict(self.errors_by_type),
                "recent": self.recent_errors[-10:],  # Last 10 errors
            },
        }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.__init__()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


class RequestTimingMiddleware:
    """
    Middleware to track request timing and metrics.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500
        error_msg = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            method = scope["method"]
            path = scope["path"]

            # Track the request
            collector = get_metrics_collector()
            collector.track_request(
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                error=error_msg,
            )
