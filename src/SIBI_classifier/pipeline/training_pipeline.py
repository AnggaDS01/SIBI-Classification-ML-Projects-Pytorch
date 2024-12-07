import sys
import inspect

from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.configuration.configuration import ConfigurationManager

from SIBI_classifier.components.data_ingestion_components.data_ingestion import DataIngestion
from SIBI_classifier.components.data_preprocessing_components.data_preprocessing import DataPreprocessing
from SIBI_classifier.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

from SIBI_classifier.utils.main_utils import display_function_info
from SIBI_classifier.logger.logging import log_manager

class TrainPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_preprocessing_config = config.get_data_preprocessing_config()

    
    def start_data_ingestion(self) -> DataIngestionConfig:
        try:
            logger = log_manager.setup_logger("DataIngestionPipelineLogger")
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Started {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            # =================================== TODO: Add data ingestion code here ===================================

            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )

            data_ingestion.initiate_data_ingestion()

            # ================================================================================================

            logger.info(f"Finished {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}\n\n")

            return self.data_ingestion_config

        except Exception as e:
            raise SIBIClassificationException(e, sys)
        
    def start_data_preprocessing(
            self,
            data_ingestion_config: DataIngestionConfig
        ) -> DataPreprocessingConfig:

        try:
            logger = log_manager.setup_logger("DataPreprocessingPipelineLogger")
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Started {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            # =================================== TODO: Add data preprocessing code here ===================================

            data_preprocessing = DataPreprocessing(
                data_ingestion_config = data_ingestion_config,
                data_preprocessing_config = self.data_preprocessing_config
            )

            data_preprocessing.initiate_data_preprocessing()

            # ================================================================================================

            logger.info(f"Finished {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}\n\n")

            return self.data_preprocessing_config

        except Exception as e:
            raise SIBIClassificationException(e, sys)

    def run_pipeline(self) -> None:
        try:
            data_ingestion_config = self.start_data_ingestion()
            self.start_data_preprocessing(data_ingestion_config)
            

            log_manager.clean_log_file()
        except Exception as e:
            raise SIBIClassificationException(e, sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()