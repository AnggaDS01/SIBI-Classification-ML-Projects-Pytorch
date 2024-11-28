import requests
import sys

from tqdm import tqdm
from pathlib import Path
from SIBI_classifier.exception import SIBIClassificationException

from SIBI_classifier.logger.logging import log_manager

def download_zip(
    url: str, 
    save_zip_file_path: Path, 
    chunk_size: int = 1024
) -> None:
    """
    Downloads a file from a given URL to the specified path.
    
    Args:
        url (str): URL of the file to download.
        save_zip_file_path (Path): The path where the file will be saved.
        chunk_size (int): The chunk size for download. Default is 1024 (1 KB).
    
    Raises:
        requests.exceptions.RequestException: If an error occurs during download.
    """
    try:
        logger = log_manager.setup_logger("download_zip_logger")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        with open(save_zip_file_path, "wb") as file, tqdm(
                desc=f"Downloading {save_zip_file_path}",
                total=total_size,
                unit='B', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                bar.update(len(chunk))

        logger.debug(f"File downloaded to {log_manager.color_text(save_zip_file_path, 'blue')}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error downloading the file: {e}")
    
    except Exception as e:
        raise SIBIClassificationException(e, sys)