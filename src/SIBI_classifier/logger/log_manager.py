import os
import re
import logging
from datetime import datetime
from colorlog import ColoredFormatter


# class SingletonMeta(type):
#     """
#     Metaclass for creating Singletons.
#     """
#     _instances = {}

#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             instance = super().__call__(*args, **kwargs)
#             cls._instances[cls] = instance
#         return cls._instances[cls]
    

class LogManager:
    def __init__(self, log_dir="logs", log_name=None):
        """
        Initialize the LogManager with the log directory and log file name.

        :param log_dir: The directory where the logs will be stored.
        :param log_name: The name of the log file. If not given, the name will be a timestamp.
        """
        self.log_dir = log_dir
        self.log_name = log_name or f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_path = os.path.join(self.log_dir, self.log_name)

        
        os.makedirs(self.log_dir, exist_ok=True)

        self.ansi_escape = re.compile(r'\033\[[0-9;]*m')

    @staticmethod
    def color_text(text, color):
        """
        Color the text using ANSI code.

        Args:
            text (str): The text to be colored.
            color (str): The color to use. The color should be a valid ANSI color code.
                The supported colors are: ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "reset"]

        Returns:
            str: The colored text.
        """
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

    def highlight_keywords(self, message, keywords, color):
        """
        Highlight keywords in the given message with a specific color.

        Args:
            message (str): The message in which to highlight keywords.
            keywords (list): A list of keywords to highlight in the message.
            color (str): The color in which to highlight the keywords.

        Returns:
            str: The message with highlighted keywords.
        """
        for keyword in keywords:
            # Replace each keyword with its colored version in the message
            message = message.replace(keyword, self.color_text(keyword, color))
        return message

    def remove_ansi_codes(self, text):
        """
        Remove ANSI codes from text.
        """
        return self.ansi_escape.sub('', text)

    def setup_logger(self, logger_name, console_output=True):
        """
        Set up a logger with handlers for both console (colored) and file (uncolored).
        
        Args:
            logger_name (str): The name of the logger.
            console_output (bool): If True, add console handler for colored output.
        
        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Create file handler for logging to a file
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if console_output:
            # Create console handler for logging to the console with color
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
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
            logger.addHandler(console_handler)

        return logger

    def clean_log_file(self):
        """
        Remove ANSI color codes from the log file.

        This function cleans the log file of ANSI color codes. This is useful for
        viewing the log file in a text editor or other program that doesn't
        support these codes.

        :return: None
        """
        with open(self.log_path, 'r+') as log_file:
            # Read all the lines in the log file
            cleaned_lines = [self.remove_ansi_codes(line) for line in log_file]

            # Go back to the beginning of the file
            log_file.seek(0)

            # Write all the cleaned lines back to the file
            log_file.writelines(cleaned_lines)

            # Truncate the file to the new size
            log_file.truncate()
        # print(f"Log file cleaned: {self.log_path}")