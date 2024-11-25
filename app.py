from src.SIBI_classifier.logger.logger_config import LogManager

# Contoh Penggunaan
if __name__ == "__main__":
    log_manager = LogManager()

    # Setup logger
    logger = log_manager.setup_logger("MainLogger")

    # Contoh log dengan warna
    logger.info(f"This is an {log_manager.color_text('informational message', 'cyan')}.")

    # Keyword yang mau di-highlight
    logger.debug("This is a debug message.")
    logger.warning("This is a warning!")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Highlight pesan dengan warna
    keywords = ["informational", "error", "warning"]
    raw_message = "This is an informational message with a warning."
    highlighted_message = log_manager.highlight_keywords(raw_message, keywords, "yellow")
    logger.info(highlighted_message)

    # Setup logger
    logger = log_manager.setup_logger("SecondLogger")

    # Contoh log dengan warna
    logger.info(f"This is an {log_manager.color_text('informational message', 'cyan')}.")

    # Keyword yang mau di-highlight
    logger.debug("This is a debug message.")
    logger.warning("This is a warning!")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Highlight pesan dengan warna
    keywords = ["informational", "error", "warning"]
    raw_message = "This is an informational message with a warning."
    highlighted_message = log_manager.highlight_keywords(raw_message, keywords, "yellow")
    logger.info(highlighted_message)

    # Bersihkan log dari kode ANSI
    log_manager.clean_log_file()