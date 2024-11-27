import sys
import re
import traceback

from colorama import Fore, Style
from SIBI_classifier.logger.logging import log_manager

def error_message_detail(error, error_detail: sys):
    """
    Formats the error message with file name, line number, and error message.

    Args:
        error (Exception): The exception that occurred.
        error_detail (sys): The sys module to retrieve traceback.

    Returns:
        str: Formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: '{file_name}', line: {line_number}, error message: '{error}'"
    return error_message

def format_traceback(error_detail: sys):
    """
    Extracts and formats the full traceback.

    Args:
        error_detail (sys): The sys module to retrieve traceback.

    Returns:
        str: Full formatted traceback as a string.
    """
    return "".join(traceback.format_exception(*error_detail.exc_info()))

class SIBIClassificationException(Exception):
    def __init__(self, error_message, error_detail):
        """
        Custom Exception class to log error details and provide meaningful debugging information.

        Args:
            error_message (Exception): The exception message.
            error_detail (sys): System info to extract traceback.
            logger (logging.Logger, optional): Logger instance for logging errors.
        """
        super().__init__(error_message)
        
        # Format error message
        self.error_message = error_message_detail(error_message, error_detail)
        self.traceback = format_traceback(error_detail)

        self.logger = log_manager.setup_logger("ErrorLogger")
        error_colored_message = self.print_colored_error()
        self.logger.error(error_colored_message) 

        self.traceback_logger = log_manager.setup_logger("TracebackLogger", console_output=False)
        self.traceback_logger.debug(self.traceback)

        log_manager.clean_log_file()

    def print_colored_error(self):
        """
        Prints the formatted error message and traceback with color to the console.
        """
        file_color = Fore.CYAN
        line_color = Fore.YELLOW
        error_color = Fore.RED
        reset_color = Style.RESET_ALL

        # Pola regex untuk mencocokkan path, nomor baris, dan pesan error
        pattern = r"script: '(.+?)', line: (\d+), error message: '(.+)'"

        # Cari kecocokan menggunakan regex
        match_pattern = re.search(pattern, self.error_message)

        if match_pattern:
            # Ekstrak hasil dari grup regex
            script_path = match_pattern.group(1)
            line_number = match_pattern.group(2)
            error_text = match_pattern.group(3)

            # Highlight parts of the error message
            error_colored_message = (
                f"Error occurred in script: {file_color}'{script_path}'{reset_color}, "
                f"line: {line_color}{line_number}{reset_color}, "
                f"error message: {error_color}'{error_text}'"
            )

        return error_colored_message
    