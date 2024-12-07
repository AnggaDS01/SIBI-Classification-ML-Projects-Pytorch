import os
import random
import torch

from PIL import Image

from torch.utils.data import Dataset
from SIBI_classifier.utils.main_utils import custom_title_print
from SIBI_classifier.logger.logging import log_manager

FILE_PATH_INFO_LOGGER = log_manager.setup_logger("file path info logger")
DATASET_INSPECT_LOGGER = log_manager.setup_logger("dataset inspect logger")
COLOR_TEXT = 'yellow'

class FilePathInfo:
    def __init__(
            self, 
            unit_file_size: str='bytes'
        ) -> None:
        
        self.unit_file_size = unit_file_size.lower()
        self.units = ['bytes', 'kb', 'mb', 'gb']
        if self.unit_file_size not in self.units:
            raise ValueError(f"Invalid unit. Choose from {self.units}.")

    def show_train_files_path_info(
            self, 
            files_path_data: list=[], 
            is_random: bool=False
        ) -> int:
        """
        Shows the file path information of the first file in the list of files.

        Args:
            files_path_data (list): A list of file paths.
            is_random (bool): If True, the first file in the list is chosen at random.

        Returns:
            int: The index of the label in the file path.
        """
        # Choose a file path at random if is_random is True
        files_path_data_plot = random.choice(files_path_data) if is_random else files_path_data[0]

        # Display the file path information and get the index of the label
        label_index = self.__display_path_info(files_path_data_plot, is_labeled=True)
        return label_index

    def show_test_files_path_info(
            self, 
            files_path_data: list=[], 
            is_random: bool=False
        ) -> None:
        """
        Shows the file path information of the first file in the list of files.

        Args:
            files_path_data (list): A list of file paths.
            is_random (bool): If True, the first file in the list is chosen at random.

        Returns:
            None
        """

        # Choose a file path at random if is_random is True
        files_path_data_plot = random.choice(files_path_data) if is_random else files_path_data[0]

        # Display the file path information
        self.__display_path_info(files_path_data_plot, is_labeled=False)

    def __display_path_info(
            self, 
            file_path: str, 
            is_labeled: bool
        ) -> int:
        """
        Displays the file path information of a file.

        Args:
            file_path (str): The file path.
            is_labeled (bool): If True, the file path is assumed to be labeled.

        Returns:
            int: The index of the label in the file path if is_labeled is True, otherwise None.
        """

        # Display the file path information
        custom_title_print(' PATH INFO ')
        FILE_PATH_INFO_LOGGER.info(f'File Path: {log_manager.color_text(file_path, COLOR_TEXT)}')

        # Split the file path into a list of strings
        split_file_path = file_path.split(os.path.sep)

        # Display the split file path
        self.__display_split_file_path(split_file_path)

        if is_labeled:
            # Get the kind data from the file path
            kind_data = split_file_path[-3]

            # Display the kind data and get the index of the label
            index_label = self.__display_kind_data_info(split_file_path, kind_data)

            # Display the file information
            self.__display_file_info(split_file_path, file_path)

            # Return the index of the label
            return index_label
        else:
            # Display the file information
            self.__display_file_info(split_file_path, file_path)

    def __display_split_file_path(
            self, 
            split_file_path: list
        ) -> None:
        """
        Displays the split file path as a list of strings and as a dictionary with the index of each string.

        The split file path is the file path split into a list of strings using the os.path.sep separator.

        The dictionary contains the file path strings as keys and a string describing the index of the string as the value.

        For example, if the file path is '/a/b/c/d.txt', the split file path will be ['a', 'b', 'c', 'd.txt'] and the dictionary will be:
        {
            'a': 'Index -> 0',
            'b': 'Index -> 1',
            'c': 'Index -> 2',
            'd.txt': 'Index -> 3'
        }
        """
        
        custom_title_print(' SPLIT FILE PATH ')
        # Display the split file path as a list of strings
        FILE_PATH_INFO_LOGGER.info(f'Split File Path: {log_manager.color_text(split_file_path, COLOR_TEXT)}')

        custom_title_print(' INDEXED PATH ')
        # Create a dictionary with the file path strings as keys and a string describing the index of the string as the value
        result = {value: f'Index -> {index}' for index, value in enumerate(split_file_path)}
        # Display the dictionary
        for key, value in result.items():
            # Display each key-value pair with the key and value colored yellow
            FILE_PATH_INFO_LOGGER.info(f'{log_manager.color_text(value, COLOR_TEXT)}: {log_manager.color_text(key, COLOR_TEXT)}')

    def __display_kind_data_info(
            self, 
            split_file_path: list, 
            kind_data: str
        ) -> int:
        """
        Displays information about the kind data in the split file path and returns the index of the label.

        Args:
            split_file_path (list): A list representing the split components of the file path.
            kind_data (str): The kind of data to locate within the split file path.

        Returns:
            int: The index of the label in the split file path.
        """
        
        # Print a custom title indicating the section for kind data index
        custom_title_print(f' KIND DATA INDEX {kind_data} ')

        # Find the index of the specified kind_data in the split file path
        index = split_file_path.index(kind_data)

        # Log the found index of kind_data using the logger
        FILE_PATH_INFO_LOGGER.info(f'Index of "{log_manager.color_text(kind_data, COLOR_TEXT)}": {log_manager.color_text(index, COLOR_TEXT)}')

        # Calculate the index of the label, which is one position after the kind_data
        index_label = index + 1

        # Print a custom title indicating the section for index label
        custom_title_print(' INDEX LABEL ')

        # Log the calculated index label
        FILE_PATH_INFO_LOGGER.info(f'Index Label: {log_manager.color_text(index_label, COLOR_TEXT)}')

        # Print a custom title indicating the section for the label
        custom_title_print(' LABEL ')

        # Log the actual label found at the calculated index in the split file path
        FILE_PATH_INFO_LOGGER.info(f'Label: {log_manager.color_text(split_file_path[index_label], COLOR_TEXT)}')

        # Return the index of the label
        return index_label

    def __display_file_info(
            self, 
            split_file_path: list, 
            file_path: str
        ) -> None:

        """
        Displays information about the file.

        Args:
            split_file_path (list): A list representing the split components of the file path.
            file_path (str): The file path.
        """
        
        # Get the file name by taking the last element of the split file path
        file_name = split_file_path[-1]
        # Print a custom title indicating the section for file name
        custom_title_print(' FILE NAME ')
        # Log the file name
        FILE_PATH_INFO_LOGGER.info(f'File Name: {log_manager.color_text(file_name, COLOR_TEXT)}')

        # Get the file extension by taking the last element of the split file path and using the os.path.splitext
        file_extension = os.path.splitext(file_name)[1]
        # Print a custom title indicating the section for file extension
        custom_title_print(' FILE EXTENSION ')
        # Log the file extension
        FILE_PATH_INFO_LOGGER.info(f'File Extension: {log_manager.color_text(file_extension, COLOR_TEXT)}')

        # Attempt to open the image file using PIL
        try:
            image = Image.open(file_path)
            # Get the size of the image
            image_size = image.size
            # Print a custom title indicating the section for image size
            custom_title_print(' IMAGE SIZE (PX)')
            # Log the image size
            FILE_PATH_INFO_LOGGER.info(f'Image Size: width={log_manager.color_text(image_size[0], COLOR_TEXT)}, height={log_manager.color_text(image_size[1], COLOR_TEXT)}')
        except IOError:
            # If the file is not an image, log a message and continue
            FILE_PATH_INFO_LOGGER.info(f'{log_manager.color_text(file_path, COLOR_TEXT)} is not an image file.')

        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)
        # Convert the file size to the specified unit
        file_size = self.__format_file_size(file_size)
        # Print a custom title indicating the section for file size
        custom_title_print(' FILE SIZE ')
        # Log the file size
        FILE_PATH_INFO_LOGGER.info(f'File Size: {log_manager.color_text(file_size, COLOR_TEXT)} {log_manager.color_text(self.unit_file_size, COLOR_TEXT)}')

    def __format_file_size(
            self, 
            size: int
        ) -> float:
        """
        Converts the file size from bytes to the specified unit.

        Args:
            size (int): The size of the file in bytes.

        Returns:
            float: The file size converted to the specified unit, rounded to four decimal places.
        """

        # Determine the unit to convert the file size to
        match self.unit_file_size:
            case 'kb':
                # Convert bytes to kilobytes
                size /= 1024
            case 'mb':
                # Convert bytes to megabytes
                size /= 1024 ** 2
            case 'gb':
                # Convert bytes to gigabytes
                size /= 1024 ** 3
            case _:
                # Default case is bytes, no conversion needed
                size = size 
        
        # Round the converted size to four decimal places
        size = round(size, 4)

        # Return the converted and rounded file size
        return size


class DataInspector:
    def __init__(
            self, 
            label_list: list=None, 
        ) -> None:
        
        self.label_list = label_list

    def _inspect_single_dataset(
            self, 
            dataset: Dataset=None, 
            dataset_name: str=None, 
            idx: int=None
        ) -> None:
        
        get_idx = random.randint(0, len(dataset)) if idx == None else idx
        image, label = dataset[get_idx]
        self._print_data_info(f"{dataset_name}_data info", image, label)

    def _print_data_info(
            self, 
            title: str=None, 
            image: torch.Tensor=None, 
            label: torch.Tensor=None
        ) -> None:
        
        custom_title_print(title)
        DATASET_INSPECT_LOGGER.info(f'shape-image: {log_manager.color_text(image.shape, COLOR_TEXT)}')
        DATASET_INSPECT_LOGGER.info(f'dtype-image: {log_manager.color_text(image.dtype, COLOR_TEXT)}')
        DATASET_INSPECT_LOGGER.info(f'max-intensity: {log_manager.color_text(image.max(), COLOR_TEXT)}')
        DATASET_INSPECT_LOGGER.info(f'min-intensity: {log_manager.color_text(image.min(), COLOR_TEXT)}')
        DATASET_INSPECT_LOGGER.info(f'label: {log_manager.color_text(label, COLOR_TEXT)} -> {log_manager.color_text(self.label_list[label], COLOR_TEXT)}')
        DATASET_INSPECT_LOGGER.info(f'label-shape: {log_manager.color_text(label.shape, COLOR_TEXT)}')
        DATASET_INSPECT_LOGGER.info(f'label-type: {log_manager.color_text(label.dtype, COLOR_TEXT)}')

    def inspect(
            self, 
            idx: int=None, 
            **datasets: Dataset
        ) -> None:
        
        for dataset_name, dataset in datasets.items():
            self._inspect_single_dataset(dataset, dataset_name, idx)