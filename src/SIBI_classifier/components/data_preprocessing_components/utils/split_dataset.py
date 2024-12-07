from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

from SIBI_classifier.logger.logging import log_manager

SPLIT_TRAIN_VALID_TEST_LOGGER = log_manager.setup_logger("split train valid test logger")
COLOR_TEXT = 'yellow'

class DatasetSplitter:
    def __init__(self):
        pass

    def split_train_valid_test(
            self, 
            dataset: Dataset=None, 
            split_ratio: tuple=None, 
            shuffle: bool=True, 
            seed: int=42
        ) -> tuple:
        """
        Split the dataset into three parts: train, validation and test.

        Args:
            dataset (Dataset): The dataset to split.
            split_ratio (tuple): A tuple of three floats representing the proportion of the dataset for each split.
                The first element is the proportion of the dataset for the training set,
                the second element is the proportion of the dataset for the validation set,
                and the third element is the proportion of the dataset for the test set.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.
            seed (int, optional): The seed for the shuffle.

        Returns:
            tuple: A tuple of three datasets: the training dataset, the validation dataset and the test dataset.
        """
        if split_ratio is None or len(split_ratio) < 2:
            raise ValueError("split_ratio must be of the form (train_ratio, val_ratio).")

        train_ratio, val_ratio = split_ratio
        test_ratio = max(1.0 - (train_ratio + val_ratio), 0)  # Ensure that the test ratio is at least 0

        total_ratio = round(sum((train_ratio, val_ratio, test_ratio)), 2)
        if total_ratio != 1.0:
            raise ValueError("[ERROR] split_ratio must sum to 1.0.")

        dataset_size = len(dataset)
        labels = [data[1] for data in dataset]  # Get all the labels in the dataset

        # Split the dataset into two parts: training set and the rest (validation and test)
        train_idx, temp_idx, _, temp_labels = train_test_split(
            list(range(dataset_size)),  # Split the indices of the dataset
            labels,  # Split the labels
            test_size=(val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else None,  # Split the dataset into two parts
            stratify=labels if labels else None,  # Use the labels to stratify the split
            random_state=seed,  # Use the given seed to shuffle the dataset
        )

        # Split the rest into two parts: validation set and test set
        if test_ratio > 0:  
            val_size_ratio = val_ratio / (val_ratio + test_ratio)  # Calculate the proportion of the validation set
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx,  # Split the indices of the rest
                temp_labels,  # Split the labels of the rest
                test_size=(1 - val_size_ratio),  # Split the rest into two parts
                stratify=temp_labels if temp_labels else None,  # Use the labels to stratify the split
                random_state=seed,  # Use the given seed to shuffle the dataset
            )
        else: 
            val_idx = temp_idx  # If there is no test set, use the rest as the validation set
            test_idx = []  # The test set is empty

        train_dataset = Subset(dataset, train_idx)  # Create a subset of the dataset for the training set
        val_dataset = Subset(dataset, val_idx)  # Create a subset of the dataset for the validation set
        test_dataset = Subset(dataset, test_idx)  # Create a subset of the dataset for the test set

        self.__display_info(
            dataset_size=dataset_size,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            test_dataset=test_dataset,
            shuffle=shuffle,
        )  # Display the information of the split

        return train_dataset, val_dataset, test_dataset  # Return the three datasets

    def __display_info(
            self, 
            dataset_size: int, 
            train_dataset: Dataset, 
            valid_dataset: Dataset, 
            test_dataset: Dataset, 
            shuffle: bool
        ) -> None:
        
        """
        Displays information about the split dataset.
        """
        train_ratio = len(train_dataset) / dataset_size
        valid_ratio = len(valid_dataset) / dataset_size
        test_ratio = len(test_dataset) / dataset_size

        SPLIT_TRAIN_VALID_TEST_LOGGER.info(f"Total number of data: {log_manager.color_text(dataset_size, COLOR_TEXT)}")
        SPLIT_TRAIN_VALID_TEST_LOGGER.info(f"Shuffle status: {log_manager.color_text(shuffle, COLOR_TEXT)}")
        SPLIT_TRAIN_VALID_TEST_LOGGER.info(f"Training Dataset: {log_manager.color_text(len(train_dataset), COLOR_TEXT)} ({log_manager.color_text(f'{train_ratio * 100:.2f}%', COLOR_TEXT)})")
        SPLIT_TRAIN_VALID_TEST_LOGGER.info(f"Validation Dataset: {log_manager.color_text(len(valid_dataset), COLOR_TEXT)} ({log_manager.color_text(f'{valid_ratio * 100:.2f}%', COLOR_TEXT)})")
        SPLIT_TRAIN_VALID_TEST_LOGGER.info(f"Test Dataset: {log_manager.color_text(len(test_dataset), COLOR_TEXT)} ({log_manager.color_text(f'{test_ratio * 100:.2f}%', COLOR_TEXT)})")