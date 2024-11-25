import logging
import os
from datetime import datetime
from from_root import from_root
from colorlog import ColoredFormatter

def setup_logger(logger_name):
    """
    Set up a logger with file and colored console handlers.

    Args:
        logger_name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Generate a timestamped log file
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    # Create directory for logs if it doesn't exist
    log_path = os.path.join(from_root(), 'logs')
    os.makedirs(log_path, exist_ok=True)
    LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # File handler (plain formatting)
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler (colored formatting)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define colored formatter
    color_formatter = ColoredFormatter(
        "%(log_color)s[ %(asctime)s ] %(name)s%(reset)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "white,bg_red",
        },
    )
    console_handler.setFormatter(color_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger