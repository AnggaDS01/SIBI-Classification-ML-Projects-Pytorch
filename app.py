from src.SIBI_classifier.logger.logger_config import setup_logger


if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("MainLogger")
    logger.info("This is an informational message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning!")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")