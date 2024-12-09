import sys
import numpy as np

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from SIBI_classifier.utils.main_utils import custom_title_print
from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.logger.logging import *

COLOR_TEXT = 'yellow'

def get_dataloader(
        dataset_train: Dataset, 
        dataset_val: Dataset, 
        batch_size: int, 
        device_mode: str, 
        num_workers: int,
    ) -> tuple:
    """
    Creates data loaders for training and validation datasets.

    Args:
        dataset_train (Dataset): The training dataset.
        dataset_val (Dataset): The validation dataset.
        batch_size (int): Number of samples per batch to load.
        device_mode (str): Mode indicating CPU or GPU usage.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    try:
        # Determine if distributed processing is used (multi-GPU setup)
        is_distributed = device_mode == "multi-gpu"
        
        # Pin memory for faster data transfer to GPU, if not using CPU
        pin_memory = True if device_mode != "cpu" else False

        if is_distributed:
            # Use a distributed sampler to partition data among multiple GPUs
            train_sampler = DistributedSampler(dataset_train)
            
            # Create a data loader for the training dataset with distributed sampling
            train_loader = DataLoader(
                dataset_train, 
                batch_size=batch_size, 
                sampler=train_sampler, 
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                worker_init_fn=lambda _: np.random.seed(42)  # Seed workers for reproducibility
            )
        else:
            # Create a data loader for the training dataset with data shuffling
            train_loader = DataLoader(
                dataset_train, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                worker_init_fn=lambda _: np.random.seed(42)  # Seed workers for reproducibility
            )

        # Create a data loader for the validation dataset without shuffling
        val_loader = DataLoader(
            dataset_val, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            worker_init_fn=lambda _: np.random.seed(42)  # Seed workers for reproducibility
        )

        # Log training dataset info
        custom_title_print("TRAIN DATASET")
        DATA_BATCHING_LOGGER.info(f"BATCH_SIZE: {log_manager.color_text(batch_size, COLOR_TEXT)}")
        DATA_BATCHING_LOGGER.info(f"Number of data: {log_manager.color_text(len(dataset_train), COLOR_TEXT)}")
        DATA_BATCHING_LOGGER.info(f"Number of data (after batch): {log_manager.color_text(len(train_loader), COLOR_TEXT)}")

        # Log validation dataset info
        custom_title_print("VALIDATION DATASET")
        DATA_BATCHING_LOGGER.info(f"BATCH_SIZE: {log_manager.color_text(batch_size, COLOR_TEXT)}")
        DATA_BATCHING_LOGGER.info(f"Number of data: {log_manager.color_text(len(dataset_val), COLOR_TEXT)}")
        DATA_BATCHING_LOGGER.info(f"Number of data (after batch): {log_manager.color_text(len(val_loader), COLOR_TEXT)}")

        return train_loader, val_loader
    
    except Exception as e:
        # Handle exceptions by raising a custom classification exception
        raise SIBIClassificationException(e, sys)
