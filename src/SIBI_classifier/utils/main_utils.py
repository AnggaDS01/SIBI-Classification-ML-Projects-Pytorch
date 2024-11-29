import os 
import inspect 
import sys
import yaml

from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.logger.logging import log_manager

create_directories_logger = log_manager.setup_logger("create_directories_logger")
custom_title_print_logger = log_manager.setup_logger("title_print_logger")

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