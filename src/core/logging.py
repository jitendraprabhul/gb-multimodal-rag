"""
Structured logging configuration using structlog.

Provides consistent, structured logging across all components
with support for both development (colored) and production (JSON) formats.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    add_timestamp: bool = True,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON logs (for production)
        add_timestamp: If True, add timestamp to log entries
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        # Production: JSON output
        shared_processors.extend([
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ])
    else:
        # Development: colored console output
        shared_processors.extend([
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ])

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance with optional initial context.

    Args:
        name: Logger name (typically __name__ of the module)
        **initial_context: Initial context values to bind to the logger

    Returns:
        A bound structlog logger instance

    Example:
        logger = get_logger(__name__, component="etl")
        logger.info("Processing document", doc_id="123")
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LoggerMixin:
    """
    Mixin class that provides a logger property for any class.

    Usage:
        class MyService(LoggerMixin):
            def process(self):
                self.logger.info("Processing started")
    """

    _logger: structlog.stdlib.BoundLogger | None = None

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger bound to this class."""
        if self._logger is None:
            self._logger = get_logger(
                self.__class__.__module__,
                component=self.__class__.__name__,
            )
        return self._logger


def log_operation(
    operation: str,
    success: bool = True,
    duration_ms: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Log an operation with standard fields.

    Args:
        operation: Name of the operation
        success: Whether the operation succeeded
        duration_ms: Duration in milliseconds
        **kwargs: Additional context
    """
    logger = get_logger("operations")
    log_data = {
        "operation": operation,
        "success": success,
        **kwargs,
    }
    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)

    if success:
        logger.info("Operation completed", **log_data)
    else:
        logger.error("Operation failed", **log_data)
