 # Logging setup

"""
Logging configuration for FairLens AI
"""
import sys
from loguru import logger
from pathlib import Path
from app.utils.constants import LOG_LEVEL, LOG_FORMAT, LOGS_DIR

# Remove default logger
logger.remove()

# Create logs directory
Path(LOGS_DIR).mkdir(exist_ok=True)

# Console logger
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True
)

# File logger
logger.add(
    f"{LOGS_DIR}/fairlens_{{time:YYYY-MM-DD}}.log",
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    rotation="1 day",
    retention="30 days",
    compression="zip"
)

def get_logger():
    """Get configured logger instance"""
    return logger