import sys
import inspect
import torch

from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.configuration.configuration import ConfigurationManager

from SIBI_classifier.components.data_ingestion_components.data_ingestion import DataIngestion
from SIBI_classifier.components.data_preprocessing_components.data_preprocessing import DataPreprocessing
from SIBI_classifier.components.model_trainer_components.model_trainer import ModelTrainer
from SIBI_classifier.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig, ModelTrainerConfig, WandbConfig)

from SIBI_classifier.utils.main_utils import display_function_info
from SIBI_classifier.logger.logging import log_manager

class TrainPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
        self.model_trainer_config = config.get_model_trainer_config()
        self.wandb_config = config.get_wandb_config()

    
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

            train_pt_datasets, valid_pt_datasets = data_preprocessing.initiate_data_preprocessing()

            # ================================================================================================

            logger.info(f"Finished {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}\n\n")

            return train_pt_datasets, valid_pt_datasets

        except Exception as e:
            raise SIBIClassificationException(e, sys)
        
    def start_model_trainer(
            self,
            train_pt_datasets: torch.utils.data.Dataset = None,
            valid_pt_datasets: torch.utils.data.Dataset = None,
        ) -> ModelTrainerConfig:

        try:
            logger = log_manager.setup_logger("ModelTrainerPipelineLogger")
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Started {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            # =================================== TODO: Add model trainer code here ===================================

            model_trainer = ModelTrainer(
                train_pt_datasets = train_pt_datasets,
                valid_pt_datasets = valid_pt_datasets,
                wandb_config = self.wandb_config,
                data_preprocessing_config = self.data_preprocessing_config,
                model_trainer_config = self.model_trainer_config
            )

            model_trainer.initiate_model_trainer()

            # ================================================================================================

            logger.info(f"Finished {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}\n\n")

            return self.model_trainer_config

        except Exception as e:
            raise SIBIClassificationException(e, sys)

    def run_pipeline(self) -> None:
        try:
            data_ingestion_config = self.start_data_ingestion()
            train_pt_datasets, valid_pt_datasets = self.start_data_preprocessing(data_ingestion_config)
            self.start_model_trainer(train_pt_datasets, valid_pt_datasets)

            log_manager.clean_log_file()
        except Exception as e:
            raise SIBIClassificationException(e, sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()