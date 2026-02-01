"""
Tests for structured logging configuration.

Covers setup_logging, get_logger, LoggerMixin, and log_operation.
"""

from unittest.mock import patch, MagicMock

import structlog

from src.core.logging import get_logger, LoggerMixin, log_operation, setup_logging


class TestSetupLogging:
    def test_default_setup(self):
        setup_logging()
        logger = structlog.get_logger()
        assert logger is not None

    def test_debug_level(self):
        setup_logging(level="DEBUG")
        logger = structlog.get_logger()
        assert logger is not None

    def test_json_format(self):
        setup_logging(json_format=True)
        logger = structlog.get_logger()
        assert logger is not None

    def test_no_timestamp(self):
        setup_logging(add_timestamp=False)
        logger = structlog.get_logger()
        assert logger is not None

    def test_all_log_levels(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(level=level)


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert logger is not None

    def test_with_context(self):
        logger = get_logger("test_module", component="test")
        assert logger is not None

    def test_without_name(self):
        logger = get_logger()
        assert logger is not None

    def test_multiple_context_values(self):
        logger = get_logger("test", component="etl", doc_id="123")
        assert logger is not None


class TestLoggerMixin:
    def test_provides_logger(self):
        class MyClass(LoggerMixin):
            pass

        obj = MyClass()
        logger = obj.logger
        assert logger is not None

    def test_logger_cached(self):
        class MyClass(LoggerMixin):
            pass

        obj = MyClass()
        logger1 = obj.logger
        logger2 = obj.logger
        assert logger1 is logger2

    def test_logger_bound_to_class(self):
        class MyService(LoggerMixin):
            pass

        obj = MyService()
        # The logger should exist and be usable
        assert obj.logger is not None

    def test_different_classes_get_different_loggers(self):
        class ClassA(LoggerMixin):
            pass

        class ClassB(LoggerMixin):
            pass

        a = ClassA()
        b = ClassB()
        # Each class should get its own logger
        assert a.logger is not b.logger

    def test_mixin_with_inheritance(self):
        class Base(LoggerMixin):
            pass

        class Child(Base):
            pass

        child = Child()
        assert child.logger is not None


class TestLogOperation:
    def test_success(self):
        # Should not raise
        log_operation("test_op", success=True, duration_ms=100.5)

    def test_failure(self):
        # Should not raise
        log_operation("test_op", success=False, duration_ms=50.0)

    def test_without_duration(self):
        log_operation("test_op", success=True)

    def test_with_extra_kwargs(self):
        log_operation("test_op", success=True, doc_id="123", chunk_count=5)
