from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    data_ingestion_dir_path: Path
    data_download_store_dir_path: Path
    data_download_store_train_dir_path: Path
    data_download_store_test_dir_path: Path
    zip_file_path: Path
    data_download_url: str

@dataclass
class DataPreprocessingConfig:
    labels_list_file_path: Path
    class_weights_file_path: Path
    image_extension_regex: str
    label_list: list
    split_ratio: tuple
    img_size: tuple
    seed: int
    brightness: tuple
    contrast: tuple
    saturation: tuple
    hue: tuple
    p: float

@dataclass
class ModelTrainerConfig:
    model_file_path: Path
    training_table_file_path: Path
    epoch_table_file_path: Path
    training_plot_file_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    loss_function: str
    metrics: list

@dataclass
class WandbConfig:
    project_name: str
    config: dict
    sweep_config: dict
    sweep_count: int