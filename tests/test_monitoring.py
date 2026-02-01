"""
Tests for monitoring and metrics.
"""

from src.api.monitoring import MetricsCollector


class TestMetricsCollector:
    """Tests for metrics collection."""

    def test_track_request(self):
        """Test tracking requests."""
        collector = MetricsCollector()

        collector.track_request(
            method="GET",
            path="/api/v1/health",
            status_code=200,
            duration_ms=10.5,
        )

        metrics = collector.get_metrics()

        assert metrics["requests"]["total"] == 1
        assert "GET /api/v1/health" in metrics["requests"]["by_endpoint"]

    def test_track_error(self):
        """Test tracking errors."""
        collector = MetricsCollector()

        collector.track_request(
            method="POST",
            path="/api/v1/ask",
            status_code=500,
            duration_ms=100.0,
            error="Internal server error",
        )

        metrics = collector.get_metrics()

        assert metrics["requests"]["errors"]["total"] == 1
        assert len(metrics["errors"]["recent"]) == 1

    def test_track_component(self):
        """Test tracking component metrics."""
        collector = MetricsCollector()

        collector.track_component("query", 50.0)
        collector.track_component("query", 75.0)
        collector.track_component("query", 100.0)

        metrics = collector.get_metrics()

        assert metrics["components"]["query"]["count"] == 3
        assert metrics["components"]["query"]["latency"]["mean"] == 75.0

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        collector = MetricsCollector()

        # Add latencies
        for i in range(100):
            collector.track_request(
                method="GET",
                path="/test",
                status_code=200,
                duration_ms=float(i),
            )

        metrics = collector.get_metrics()
        latency = metrics["requests"]["latency"]["GET /test"]

        assert latency["p50"] > 40
        assert latency["p95"] > 90
        assert latency["p99"] > 95
