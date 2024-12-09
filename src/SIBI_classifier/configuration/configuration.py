import torch

from SIBI_classifier.constant import *
from pathlib import Path
from SIBI_classifier.utils.main_utils import read_yaml, create_directories
from SIBI_classifier.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig, ModelTrainerConfig, WandbConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.ARTIFACTS_ROOT_DIR])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.DATA_INGESTION
        create_directories([config.DATA_DOWNLOAD_STORE_DIR_PATH])

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir_path = Path(config.DATA_INGESTION_DIR_PATH),
            data_download_store_dir_path = Path(config.DATA_DOWNLOAD_STORE_DIR_PATH),
            data_download_store_train_dir_path = Path(config.DATA_DOWNLOAD_STORE_TRAIN_DIR_PATH),
            data_download_store_test_dir_path = Path(config.DATA_DOWNLOAD_STORE_TEST_DIR_PATH),
            zip_file_path = Path(config.ZIP_FILE_PATH),
            data_download_url = config.DATA_DOWNLOAD_URL,
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataIngestionConfig:
        config = self.config.DATA_PREPROCESSING
        # create_directories([config.OBJECTS_DIR_PATH])

        data_preprocessing_config = DataPreprocessingConfig(
            labels_list_file_path = Path(config.LABELS_LIST_FILE_PATH),
            class_weights_file_path = Path(config.CLASS_WEIGHTS_FILE_PATH),
            image_extension_regex = Path(config.IMAGE_EXTENSION_REGEX),
            label_list = self.params.LABEL_LIST,
            split_ratio = self.params.SPLIT_RATIO,
            seed = self.params.SEED,
            img_size = self.params.IMAGE_SIZE,
            mean = self.params.MEAN,
            std = self.params.STD,
            brightness = self.params.BRIGHTNESS,
            contrast = self.params.CONTRAST,
            saturation = self.params.SATURATION,
            hue = self.params.HUE,
            p = self.params.PROB,
        )

        return data_preprocessing_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.MODEL_TRAINING
        
        create_directories([config.MODEL_DIR_PATH])
        # create_directories([config.REPORTS_DIR_PATH])

        model_trainer_config = ModelTrainerConfig(
            model_file_path = Path(config.MODEL_FILE_PATH),
            training_table_file_path = Path(config.TRAINING_TABLE_FILE_PATH),
            epoch_table_file_path = Path(config.EPOCH_TABLE_FILE_PATH),
            training_plot_file_path = Path(config.TRAINING_PLOT_FILE_PATH),
            batch_size = self.params.BATCH_SIZE,
            epochs = self.params.EPOCHS,
            learning_rate = self.params.LEARNING_RATE,
            criterion = self.params.CRITERION,
        )

        return model_trainer_config
    
    def get_wandb_config(self) -> WandbConfig:
        config = self.config.WANDB

        # Mapping dari nama ke kelas
        criterion_map = {
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
            "MSELoss": torch.nn.MSELoss,
        }

        optimizer_map = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }

        config_dicts = {
            "epochs": self.params.EPOCHS,
            "batch_size": self.params.BATCH_SIZE,
            "learning_rate": self.params.LEARNING_RATE,
            "criterion": criterion_map[self.params.CRITERION],
            "optimizer": optimizer_map[self.params.OPTIMIZER],
            "architecture": self.params.MODEL_NAME,
            "dataset": self.params.DATASET_NAME,
        }

        project = config.PROJECT_NAME
        sweep_config = config.SWEEP_CONFIG
        sweep_count = config.SWEEP_COUNT
        
        wandb_config = WandbConfig(
            project_name = project,
            config = config_dicts,
            sweep_config = sweep_config,
            sweep_count = sweep_count
        )

        return wandb_config
    
if __name__ == '__main__':
    config = ConfigurationManager()
    get_config = config.get_wandb_config()

    print(get_config.config['optimizer'])