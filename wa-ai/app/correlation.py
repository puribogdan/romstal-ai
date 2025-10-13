import logging
import uuid
import functools
import asyncio
from typing import Optional, Callable

# Get logger for this module
logger = logging.getLogger(__name__)

# Global correlation ID storage
_correlation_id_context = {'current': 'NO-CORR-ID'}

def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())

def get_correlation_id() -> str:
    """Get the current correlation ID from the logging context."""
    return _correlation_id_context.get('current', 'NO-CORR-ID')

def set_correlation_id(correlation_id: str):
    """Set the correlation ID in the logging context."""
    _correlation_id_context['current'] = correlation_id

def log_with_correlation(logger_method: Callable, msg: str, correlation_id: Optional[str] = None, **kwargs):
    """Log a message with correlation ID context."""
    if correlation_id:
        set_correlation_id(correlation_id)
    logger_method(f"[{correlation_id or get_correlation_id()}] {msg}", **kwargs)

# Enhanced exception logging decorator
def log_exceptions(correlation_id: Optional[str] = None):
    """Decorator to log exceptions with full tracebacks and correlation ID."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                corr_id = correlation_id or getattr(func, '_correlation_id', generate_correlation_id())
                logger.exception(f"Exception in {func.__name__} with correlation_id {corr_id}: {type(e).__name__}: {e}",
                               extra={'correlation_id': corr_id})
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                corr_id = correlation_id or getattr(func, '_correlation_id', generate_correlation_id())
                logger.exception(f"Exception in {func.__name__} with correlation_id {corr_id}: {type(e).__name__}: {e}",
                               extra={'correlation_id': corr_id})
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Context manager for correlation ID tracking
class CorrelationContext:
    """Context manager to set correlation ID for a block of operations."""
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.previous_id = None

    def __enter__(self):
        self.previous_id = get_correlation_id()
        set_correlation_id(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_id:
            set_correlation_id(self.previous_id)
        else:
            # Clear the correlation ID if there wasn't one before
            _correlation_id_context['current'] = 'NO-CORR-ID'