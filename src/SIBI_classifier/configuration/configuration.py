from SIBI_classifier.constant import *
from pathlib import Path
from SIBI_classifier.utils.main_utils import read_yaml, create_directories
from SIBI_classifier.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

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
        create_directories([config.OBJECTS_DIR_PATH])

        data_preprocessing_config = DataPreprocessingConfig(
            labels_list_file_path = Path(config.LABELS_LIST_FILE_PATH),
            class_weights_file_path = Path(config.CLASS_WEIGHTS_FILE_PATH),
            image_extension_regex = Path(config.IMAGE_EXTENSION_REGEX),
            label_list = self.params.LABEL_LIST,
            split_ratio = self.params.SPLIT_RATIO,
            img_size = self.params.IMAGE_SIZE,
            seed = self.params.SEED
        )

        return data_preprocessing_config
    
# if __name__ == '__main__':
#     config = ConfigurationManager()
#     get_config = config.get_model_evaluation_config()

#     print(get_config)