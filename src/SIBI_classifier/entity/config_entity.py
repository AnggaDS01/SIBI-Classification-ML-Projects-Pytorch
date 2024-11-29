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