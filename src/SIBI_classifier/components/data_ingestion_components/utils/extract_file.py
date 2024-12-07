import sys
import zipfile

from pathlib import Path
from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.logger.logging import log_manager

EXTRACT_ZIP_LOGGER = log_manager.setup_logger("extract zip logger")

def extract_zip(
    zip_file_path: Path, 
    extract_dir: Path, 
    is_file_removed: bool = True
) -> None:
    """
    Extracts a zip file to a specified directory.

    Args:
        zip_file_path (Path): The path to the zip file.
        extract_dir (Path): The directory where files will be extracted.
        is_file_removed (bool): Delete the zip file after extraction if True.
    
    Raises:
        zipfile.BadZipFile: If the file is not a valid zip file.
    """
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            EXTRACT_ZIP_LOGGER.debug(f"Files extracted to {log_manager.color_text(extract_dir, 'green')}")

        # Remove zip file if specified
        if is_file_removed and zip_file_path.exists():
            zip_file_path.unlink()
            EXTRACT_ZIP_LOGGER.debug("Downloaded zip file removed.")

    except zipfile.BadZipFile:
        raise Exception("Error: The downloaded file is not a valid zip file.")
    
    except Exception as e:
        raise SIBIClassificationException(e, sys)