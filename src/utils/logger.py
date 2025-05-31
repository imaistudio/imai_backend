import sys
from pathlib import Path
from loguru import logger
from typing import Optional
from .config import settings

def setup_logger(
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    rotation: str = "500 MB",
    retention: str = "10 days"
) -> None:
    """
    Configure the logger with the specified settings.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if log_file is specified
    if log_file:
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

# Initialize logger with settings
setup_logger(
    log_file=settings.log.log_file,
    log_level=settings.log.log_level
)

# Export logger for easy access
__all__ = ['logger'] 