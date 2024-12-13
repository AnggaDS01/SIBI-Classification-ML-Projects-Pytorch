import sys
import inspect
import torch
import wandb
import pandas as pd

from torchvision import models
from torchinfo import summary

from SIBI_classifier.utils.main_utils import display_function_info, load_object
from SIBI_classifier.logger.logging import log_manager
from SIBI_classifier.components.model_trainer_components.utils.setup_device_usage import get_device
from SIBI_classifier.components.model_trainer_components.utils.model_fit import fit

from SIBI_classifier.ml.model import TransferLearningModel
from SIBI_classifier.entity.config_entity import (ModelTrainerConfig, DataPreprocessingConfig, WandbConfig)
from SIBI_classifier.exception import SIBIClassificationException


class ModelTrainer:
    def __init__(
            self, 
            train_pt_datasets: torch.utils.data.Dataset = None,
            valid_pt_datasets: torch.utils.data.Dataset = None,
            data_preprocessing_config: DataPreprocessingConfig = DataPreprocessingConfig,
            wandb_config: WandbConfig = WandbConfig,
            model_trainer_config: ModelTrainerConfig = ModelTrainerConfig
        ):

        try:
            self.train_pt_datasets = train_pt_datasets
            self.valid_pt_datasets = valid_pt_datasets
            self.data_preprocessing_config = data_preprocessing_config
            self.wandb_config = wandb_config
            self.model_trainer_config = model_trainer_config

        except Exception as e:
           raise SIBIClassificationException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerConfig:
        try:
            logger = log_manager.setup_logger("ModelTrainerLogger")
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            logger.info(f"Entered {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")

            # =================================== TODO: Add Model Trainer code here ===================================
            wandb.init(
                project=self.wandb_config.project_name,
                config=self.wandb_config.config
            )
            config = wandb.config

            logger.info(f"Loading classes weights from file: {log_manager.color_text(self.data_preprocessing_config.class_weights_file_path, 'cyan')}")
            train_class_weights = load_object(file_path=self.data_preprocessing_config.class_weights_file_path)

            logger.info("Loading pre-trained base model...")
            # Pre-trained base model
            base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

            # Freeze the base model layers
            for param in base_model.parameters():
                param.requires_grad = False

            logger.info("Instantiating model...")
            # Instantiate the model
            model = TransferLearningModel(base_model, self.data_preprocessing_config.label_list)
            device_mode, num_gpus = get_device()

            logger.info(f"Model Architecture: \n{summary(model, input_size=(self.model_trainer_config.batch_size, 3, *self.data_preprocessing_config.img_size[:2]))}")

            history = fit(
                model,
                self.train_pt_datasets,
                self.valid_pt_datasets,
                self.data_preprocessing_config.label_list,
                train_class_weights,
                config.epochs,
                config.batch_size,
                self.wandb_config.config['criterion'],
                self.wandb_config.config['optimizer'],
                config.learning_rate,
                device_mode,
                num_gpus,
                model_file_path=self.model_trainer_config.model_file_path
            )

            # Tabel epoch setelah pelatihan selesai
            epoch_df = pd.DataFrame(history, 
                columns=[
                    "Epoch", 
                    "Train Accuracy", 
                    "Valid Accuracy", 
                    "Train Loss", 
                    "Valid Loss", 
                    "Epoch Time (s)"
                ]
            )
            wandb.log({"epoch_table": wandb.Table(dataframe=epoch_df)})
            wandb.finish()

            # ================================================================================================

            # Return the configuration used to ingest the data
            logger.info(f"Exited {log_manager.color_text(function_name, 'yellow')} method of {log_manager.color_text(class_name, 'yellow')} class in {log_manager.color_text(file_name, 'cyan')}")
            logger.info(f"{class_name} config: {log_manager.color_text(self.model_trainer_config, 'magenta')}")

            return self.model_trainer_config

        except Exception as e:
            raise SIBIClassificationException(e, sys)