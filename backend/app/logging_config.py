"""
Logging Configuration Module

Provides centralized logging configuration for the sentiment analysis pipeline.
Handles both file and console logging with appropriate formatting and rotation.

Usage:
    from app.logging_config import setup_logging, get_logger
    
    # Setup logging once at application start
    setup_logging()
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default log directory
DEFAULT_LOG_DIR = Path("logs")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """
    Setup centralized logging configuration.
    
    Creates both console and file handlers with appropriate formatting.
    File logs are stored with timestamp in the filename for easy tracking.
    
    Args:
        log_dir: Directory for log files (default: ./logs)
        log_level: Root logger level (default: INFO)
        console_level: Console handler level (default: INFO)
        file_level: File handler level (default: DEBUG)
    
    Example:
        >>> setup_logging()  # Use defaults
        >>> setup_logging(log_level="DEBUG")  # More verbose
    """
    # Use default log directory if not specified
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - general log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sentiment_pipeline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Error-only file handler
    error_log_file = log_dir / f"errors_{timestamp}.log"
    error_handler = logging.FileHandler(error_log_file, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    root_logger.info(f"Logging initialized - Log file: {log_file}")
    root_logger.info(f"Error log file: {error_log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exception: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback and optional context.
    
    Args:
        logger: Logger instance to use
        exception: Exception to log
        context: Optional context string describing where/why the error occurred
    
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, e, "During batch processing")
    """
    if context:
        logger.error(f"{context}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"Exception occurred: {str(exception)}", exc_info=True)


class FailedItemsTracker:
    """
    Tracks failed items during processing and can save them to a JSON file.
    
    Useful for tracking records that failed sentiment analysis, tokenization,
    or any other processing step.
    
    Attributes:
        failed_items: List of failed item dictionaries
        
    Example:
        >>> tracker = FailedItemsTracker()
        >>> tracker.add_failure("text content", "TokenizationError", "Text too long")
        >>> tracker.save("data/failed_items.json")
    """
    
    def __init__(self):
        """Initialize an empty failed items tracker."""
        self.failed_items = []
    
    def add_failure(
        self,
        item: str,
        error_type: str,
        error_message: str,
        additional_info: Optional[dict] = None
    ) -> None:
        """
        Add a failed item to the tracker.
        
        Args:
            item: The item that failed (text, ID, etc.)
            error_type: Type of error (e.g., 'TokenizationError', 'ValueError')
            error_message: Detailed error message
            additional_info: Optional dictionary with additional context
        """
        failure_record = {
            "item": item[:500] if isinstance(item, str) else str(item),  # Truncate long texts
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if additional_info:
            failure_record["additional_info"] = additional_info
        
        self.failed_items.append(failure_record)
    
    def save(self, output_file: Path) -> int:
        """
        Save failed items to a JSON file.
        
        Args:
            output_file: Path to output JSON file
        
        Returns:
            Number of failed items saved
        """
        import json
        
        if not self.failed_items:
            # No failures - don't create file
            return 0
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.failed_items, f, indent=2, ensure_ascii=False)
        
        return len(self.failed_items)
    
    def count(self) -> int:
        """Get the number of failed items."""
        return len(self.failed_items)
    
    def clear(self) -> None:
        """Clear all failed items."""
        self.failed_items.clear()
