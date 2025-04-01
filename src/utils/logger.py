# src/utils/logger.py

import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file: str = None):
    """
    Configures the logging settings.

    Args:
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        log_file (str, optional): Path to the log file. If None, logs are not written to a file.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define log format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)  # Ensure console handler uses the same level
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)