import re
import logging
from datetime import datetime
import os
from colorlog import ColoredFormatter

# Generate a timestamped log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Create directory for logs if it doesn't exist
log_path = os.path.join(os.getcwd(), 'logs')  # Replace `from_root()` with current directory
os.makedirs(log_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

def color_text(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"


def highlight_keywords(message, keywords, color):
    for keyword in keywords:
        message = message.replace(keyword, color_text(keyword, color))
    return message


def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return ansi_escape.sub('', text)


def setup_logger(logger_name):
    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # File handler (plain formatting, no ANSI codes)
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)

    # Formatter that removes ANSI codes
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

def clean_log_file(log_file_path=LOG_FILE_PATH):
    ansi_escape = re.compile(r'\033\[[0-9;]*m')

    with open(log_file_path, 'r+') as log_file:
        cleaned_lines = [ansi_escape.sub('', line) for line in log_file]
        log_file.seek(0) 
        log_file.writelines(cleaned_lines)
        log_file.truncate()

    print(f"Log file cleaned")