import os 
import inspect 
import sys
import yaml
import dill

from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.logger.logging import log_manager

create_directories_logger = log_manager.setup_logger("create_directories_logger")
custom_title_print_logger = log_manager.setup_logger("title_print_logger")
save_object_logger = log_manager.setup_logger("save object logger")
load_object_logger = log_manager.setup_logger("load object logger")

def read_yaml(path_to_yaml: Path=None) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise SIBIClassificationException(e, sys)

def create_directories(
        path_to_directories: list=[],
    ) -> None:

    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            create_directories_logger.info(f"created directory at: {log_manager.color_text(path, 'cyan')}")

    except Exception as e:
        raise SIBIClassificationException(e, sys)
    
def display_function_info(frame) -> tuple:
    """
    Retrieves the name of the function, class, and file location 
    of the calling function for display purposes.

    Args:
        frame (frame): A frame object representing the function 
                       or method to display information for.

    Returns:
        tuple: A tuple containing the function name, class name 
               (or None if not in a class), and file name.
    """
    try:
        # Retrieve the function's name and the file in which it is defined
        function_name = frame.f_code.co_name
        file_name = inspect.getfile(frame)
        
        # Check if there is a class context and retrieve class name if exists
        class_name = frame.f_locals.get('self')
        class_name = class_name.__class__.__name__ if class_name else None
        
        return function_name, class_name, file_name

    except Exception as e:
        raise SIBIClassificationException(e, sys)
    

def custom_title_print(
        title: str='Title', 
        n_strip: int=50
    ) -> None:

    """
    Mencetak judul yang disesuaikan dengan garis pembatas di atas dan di bawah judul.

    Args:
        title (str): Judul yang ingin ditampilkan.
        n_strip (int): Jumlah karakter '=' untuk membuat garis pembatas. Default adalah 80.

    Returns:
        None
    """

    try:
        title = f"{title.upper().center(n_strip, '=')}"
        custom_title_print_logger.info(title)

    except Exception as e:
        raise SIBIClassificationException(e, sys)

def save_object(
        file_path: str=None,  # Path to save the object
        obj: object=None  # Object to be saved
    ) -> None:

    """
    Save an object to a binary file using dill library.

    Args:
        file_path (str): Path to save the object.
        obj (object): Object to be saved.

    Returns:
        None
    """

    try:
        # Check if file already exists
        if os.path.exists(file_path): 
            # If file exists, skip saving and log a message
            save_object_logger.info(f"File '{log_manager.color_text(file_path, 'cyan')}' already exists. Skipping saving.")
            return 

        # Open the file in binary write mode
        with open(file_path, 'wb') as file_obj:
            # Use dill library to dump the object to the file
            dill.dump(obj, file_obj) 
        
        # Log a message to indicate that the object has been saved
        save_object_logger.info(f"Object saved to {log_manager.color_text(file_path, 'cyan')}")

    except Exception as e:
        # If any error occurs, raise a SIBIClassificationException
        raise SIBIClassificationException(e, sys)
    

def load_object(file_path: str=None) -> object:
    """
    Load an object from a binary file using dill library.

    Args:
        file_path (str): Path to the binary file containing the object.

    Returns:
        object: The loaded object, or None if the file does not exist.
    """
    try:
        # Open the file in binary read mode
        with open(file_path, 'rb') as file_obj:
            # Use dill library to load the object from the file
            return dill.load(file_obj)
    except FileNotFoundError:
        # If the file does not exist, log a message and return None
        load_object_logger.info(f"File not found: {log_manager.color_text(file_path, 'cyan')}")
        return None
    except Exception as e:
        # If any other error occurs, raise a SIBIClassificationException
        raise SIBIClassificationException(e, sys)
