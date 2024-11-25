import os
import re
import logging
from datetime import datetime
from colorlog import ColoredFormatter


class LogManager:
    def __init__(self, log_dir="logs", log_name=None):
        """
        Inisialisasi LogManager dengan direktori log dan nama file log.

        :param log_dir: Direktori tempat log akan disimpan.
        :param log_name: Nama file log. Jika tidak diberikan, nama akan berupa timestamp.
        """
        self.log_dir = log_dir
        self.log_name = log_name or f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_path = os.path.join(self.log_dir, self.log_name)

        # Pastikan direktori log tersedia
        os.makedirs(self.log_dir, exist_ok=True)

        # Regex untuk menghapus kode ANSI
        self.ansi_escape = re.compile(r'\033\[[0-9;]*m')

    @staticmethod
    def color_text(text, color):
        """
        Memberi warna pada teks menggunakan kode ANSI.
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
        Menyoroti kata kunci dalam pesan dengan warna tertentu.
        """
        for keyword in keywords:
            message = message.replace(keyword, self.color_text(keyword, color))
        return message

    def remove_ansi_codes(self, text):
        """
        Menghapus kode ANSI dari teks.
        """
        return self.ansi_escape.sub('', text)

    def setup_logger(self, logger_name):
        """
        Menyiapkan logger dengan handler untuk konsol (berwarna) dan file (tanpa warna).
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # File handler (tanpa warna)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Console handler (berwarna)
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

        # Tambahkan handler ke logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def clean_log_file(self):
        """
        Membersihkan file log dari kode warna ANSI.
        """
        with open(self.log_path, 'r+') as log_file:
            cleaned_lines = [self.remove_ansi_codes(line) for line in log_file]
            log_file.seek(0)
            log_file.writelines(cleaned_lines)
            log_file.truncate()
        print(f"Log file cleaned: {self.log_path}")


