from src.SIBI_classifier.logger.logger_config import setup_logger, color_text, highlight_keywords


if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("MainLogger")
    logger.info(f"This is an {color_text('informational message', 'cyan')}.")

    # Keyword yang mau di-highlight
    logger.debug("This is a debug message.")
    logger.warning("This is a warning!")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Log message dengan highlight
    keywords = ["informational", "error", "warning"]
    raw_message = "This is an informational message with a warning."
    highlighted_message = highlight_keywords(raw_message, keywords, "yellow")
    logger.info(highlighted_message)