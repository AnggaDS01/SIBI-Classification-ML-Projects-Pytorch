import sys
import inspect

from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.entity.config_entity import DataIngestionConfig

from SIBI_classifier.components.data_ingestion_components.utils.download_file import download_zip
from SIBI_classifier.components.data_ingestion_components.utils.extract_file import extract_zip
from SIBI_classifier.utils.main_utils import display_function_info
from SIBI_classifier.logger.logging import log_manager

class DataIngestion:
    def __init__(
            self, 
            data_ingestion_config: DataIngestionConfig = DataIngestionConfig
        ):

        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
           raise SIBIClassificationException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionConfig:
        """
        Initiates the data ingestion process by downloading the dataset from the given URL, extracting the dataset from the downloaded zip file, and returning the configuration used to ingest the data.

        Returns:
            DataIngestionConfig: The configuration used to ingest the data.
        """
        try:
            logger = log_manager.setup_logger("DataIngestionLogger")
            
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Entered {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            # =================================== TODO: Add data ingestion code here ===================================

            # Download the dataset from the given URL
            logger.info(f"Getting the data from URL: {log_manager.color_text(self.data_ingestion_config.data_download_url, 'blue')}")

            # Download the zip file to the specified path
            download_zip(
                url=self.data_ingestion_config.data_download_url,
                save_zip_file_path=self.data_ingestion_config.zip_file_path,
            )

            # Extract the dataset from the downloaded zip file
            logger.info(f"Extracting the dataset from the downloaded zip file: {log_manager.color_text(self.data_ingestion_config.zip_file_path, 'cyan')}")

            # Extract the dataset to the specified directory
            extract_zip(
                zip_file_path=self.data_ingestion_config.zip_file_path,
                extract_dir=self.data_ingestion_config.data_download_store_dir_path
            )

            logger.info(f"Got the data from URL: {log_manager.color_text(self.data_ingestion_config.data_download_url, 'blue')}")

            # ================================================================================================

            # Return the configuration used to ingest the data
            logger.info(f"Exited {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")
            logger.info(f"{class_name} config: {log_manager.color_text(self.data_ingestion_config, 'magenta')}")

            return self.data_ingestion_config

        except Exception as e:
            raise SIBIClassificationException(e, sys)