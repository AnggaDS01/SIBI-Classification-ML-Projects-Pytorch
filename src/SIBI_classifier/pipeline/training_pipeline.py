import sys
import inspect
from SIBI_classifier.logger.logging import log_manager
from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.configuration.configuration import ConfigurationManager
from SIBI_classifier.components.data_ingestion_components.data_ingestion import DataIngestion

from SIBI_classifier.entity.config_entity import (DataIngestionConfig)
from SIBI_classifier.utils.main_utils import (display_function_info)

class TrainPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()

    
    def start_data_ingestion(self) -> DataIngestionConfig:
        try:
            logger = log_manager.setup_logger("DataIngestionPipelineLogger")
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Started {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )

            data_ingestion.initiate_data_ingestion()

            logger.info(f"Finished {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}\n\n")

            return self.data_ingestion_config

        except Exception as e:
            raise SIBIClassificationException(e, sys)

    def run_pipeline(self) -> None:
        try:
            self.start_data_ingestion()
            

            log_manager.clean_log_file()
        except Exception as e:
            raise SIBIClassificationException(e, sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()