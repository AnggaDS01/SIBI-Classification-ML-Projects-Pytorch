from SIBI_classifier.logger.log_manager import LogManager

# global logger 
log_manager = LogManager()

DOWNLOAD_ZIP_LOGGER = log_manager.setup_logger("download zip logger")
EXTRACT_ZIP_LOGGER = log_manager.setup_logger("extract zip logger")

CLASS_DISTRIBUTION_LOGGER = log_manager.setup_logger("class distribution logger")
COLLECT_AND_COMBINE_IMAGES_LOGGER = log_manager.setup_logger("collect and combine images logger")
IMAGE_PROCESSING_LOGGER = log_manager.setup_logger("image processing logger")
FILE_PATH_INFO_LOGGER = log_manager.setup_logger("file path info logger")
DATASET_INSPECT_LOGGER = log_manager.setup_logger("dataset inspect logger")
SPLIT_TRAIN_VALID_TEST_LOGGER = log_manager.setup_logger("split train valid test logger")

DATA_BATCHING_LOGGER = log_manager.setup_logger("data batching logger")
MODEL_TRAINING_LOGGER = log_manager.setup_logger("model training logger")
SETUP_MODEL_LOGGER = log_manager.setup_logger("setup model logger")