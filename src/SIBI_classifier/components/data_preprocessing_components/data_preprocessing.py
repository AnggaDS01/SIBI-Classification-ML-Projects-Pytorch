import sys
import inspect
import torchvision.transforms as transforms

from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

from SIBI_classifier.components.data_preprocessing_components.utils.calculate_distribution_class import calculate_class_distribution_torch, print_class_distribution
from SIBI_classifier.components.data_preprocessing_components.utils.set_seed import set_seed
from SIBI_classifier.components.data_preprocessing_components.utils.get_paths import collect_and_combine_images
from SIBI_classifier.utils.main_utils import display_function_info, custom_title_print, save_object
from SIBI_classifier.logger.logging import log_manager
from SIBI_classifier.components.data_preprocessing_components.utils.split_dataset import DatasetSplitter
from SIBI_classifier.components.data_preprocessing_components.utils.show_data_info import FilePathInfo, DataInspector
from SIBI_classifier.components.data_preprocessing_components.utils.processing_dataset import ImageDataset, ConvertPathsToTensor
from SIBI_classifier.components.data_preprocessing_components.utils.image_processing import ImageProcessor
from SIBI_classifier.components.data_preprocessing_components.utils.image_processing import ImageProcessor

class DataPreprocessing:
    def __init__(
            self, 
            data_ingestion_config: DataIngestionConfig = DataIngestionConfig,
            data_preprocessing_config: DataPreprocessingConfig = DataPreprocessingConfig
        ):

        try:
            self.data_ingestion_config = data_ingestion_config
            self.data_preprocessing_config = data_preprocessing_config

        except Exception as e:
           raise SIBIClassificationException(e, sys)

    def initiate_data_preprocessing(self):
        try:
            logger = log_manager.setup_logger("DataPreprocessingLogger")
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Entered {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            # =================================== TODO: Add data processisng code here ===================================
            # Setting a seed for reproducibility
            logger.info(f"Setting random a seed: {log_manager.color_text(self.data_preprocessing_config.seed, 'yellow')} for reproducibility...")
            set_seed(self.data_preprocessing_config.seed)

            logger.info("Collecting and combining images from training and validation folders...")
            all_images_paths = collect_and_combine_images(
                classes = self.data_preprocessing_config.label_list,
                train_path  = self.data_ingestion_config.data_download_store_train_dir_path,
                pattern_regex = self.data_preprocessing_config.image_extension_regex,
                seed = self.data_preprocessing_config.seed
            )

            logger.info("Displaying file path information...")
            file_info  = FilePathInfo(unit_file_size='kb')
            label_index = file_info.show_train_files_path_info(all_images_paths, is_random=True)

            logger.info("Creating PyTorch dataset...")
            pytorch_paths = ImageDataset(all_images_paths, label_index)

            logger.info("Splitting dataset into train and validation sets...")
            splitter = DatasetSplitter()
            train_pt_paths, val_pt_paths, _ = splitter.split_train_valid_test(
                pytorch_paths, 
                split_ratio=self.data_preprocessing_config.split_ratio,
                shuffle=True,
                seed=self.data_preprocessing_config.seed
            )

            logger.info("Calculating class distribution...")
            train_class_distribution, train_class_weights = calculate_class_distribution_torch(
                dataset=train_pt_paths, 
                class_labels=self.data_preprocessing_config.label_list,
                class_weights_cvt_to_dict=False
            )
            custom_title_print("Class distribution on Train set:")
            print_class_distribution(train_class_distribution)

            valid_class_distribution, _ = calculate_class_distribution_torch(
                dataset=val_pt_paths, 
                class_labels=self.data_preprocessing_config.label_list
            )
            custom_title_print("Class distribution on Valid set:")
            print_class_distribution(valid_class_distribution)

            logger.info(f"Creating image processor...")
            image_processor = ImageProcessor(
                image_size=self.data_preprocessing_config.img_size[:2],
                mean=self.data_preprocessing_config.mean, 
                std=self.data_preprocessing_config.std,
                brightness=self.data_preprocessing_config.brightness, 
                contrast=self.data_preprocessing_config.contrast,
                saturation=self.data_preprocessing_config.saturation,
                hue=self.data_preprocessing_config.hue,
                p=self.data_preprocessing_config.p,
            )

            logger.info("Preprocessing training dataset...")
            custom_title_print("Train transforms:")
            train_transforms = transforms.Compose([
                image_processor.image_to_PIL(),
                image_processor.image_resizing(),
                image_processor.color_jitter(),
                image_processor.hFlip(),
                image_processor.vFlip(),
                image_processor.image_to_tensor(),
            ])

            train_pt_datasets = ConvertPathsToTensor(
                datasets=train_pt_paths,
                label_list=self.data_preprocessing_config.label_list,
                seed=self.data_preprocessing_config.seed,
                transform=train_transforms
            )

            logger.info("Preprocessing validation dataset...")
            custom_title_print("Validation transforms:")
            valid_transforms = transforms.Compose([
                image_processor.image_to_PIL(),
                image_processor.image_resizing(),
                image_processor.image_to_tensor(),
            ])

            valid_pt_datasets = ConvertPathsToTensor(
                datasets=val_pt_paths,
                label_list=self.data_preprocessing_config.label_list,
                seed=self.data_preprocessing_config.seed,
                transform=valid_transforms
            )

            logger.info("Inspecting datasets...")
            inspector = DataInspector(
                label_list = self.data_preprocessing_config.label_list,
            )

            inspector.inspect(
                train_dataset=train_pt_datasets,
                valid_dataset=valid_pt_datasets,
            )

            save_object(
                file_path=self.data_preprocessing_config.class_weights_file_path,
                obj=train_class_weights
            )

            # ================================================================================================

            # Return the configuration used to ingest the data
            logger.info(f"Exited {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")
            logger.info(f"{class_name} config: {log_manager.color_text(self.data_ingestion_config, 'magenta')}")

            return train_pt_datasets, valid_pt_datasets

        except Exception as e:
            raise SIBIClassificationException(e, sys)