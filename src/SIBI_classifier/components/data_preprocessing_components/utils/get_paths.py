import re
import random
import sys

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.utils.main_utils import custom_title_print
from SIBI_classifier.logger.logging import log_manager

COLLECT_AND_COMBINE_IMAGES_LOGGER = log_manager.setup_logger("collect and combine images logger")
COLOR_TEXT = "yellow"

def collect_images_with_regex_and_count(
		path: str,
		folder_classes: list,
		extensions_pattern: str
	) -> dict:
	"""
	Collects images from specified directories that match a given file extension pattern.

	Args:
		path (str): The root directory path containing the folder classes.
		folder_classes (list): List of folder names representing different classes.
		extensions_pattern (str): Regex pattern to match file extensions.

	Returns:
		dict: A dictionary where keys are folder classes and values are lists of image paths.
	"""
	
	try:
		# Pre-allocate the dictionary with the folder classes as keys
		image_paths = {folder_class: [] for folder_class in folder_classes}
		
		# Compile the regex pattern for matching file extensions
		pattern = re.compile(str(extensions_pattern), re.IGNORECASE)

		for folder_class in folder_classes:
			# Get the full path of the folder class
			folder_path = Path(path) / folder_class

			# Search the folder and subfolders for files with the given extensions
			for file_path in folder_path.rglob("*"):
				# If the pattern matches the file extension, add the file path to the dictionary
				if pattern.search(file_path.suffix):
					image_paths[folder_class].append(file_path)

			# Log the number of files found in the class
			COLLECT_AND_COMBINE_IMAGES_LOGGER.info(
				f"Collecting from class {log_manager.color_text(folder_class, COLOR_TEXT)}: {log_manager.color_text(len(image_paths[folder_class]), COLOR_TEXT)} paths found."
			)

		# Return the dictionary of file paths
		return image_paths

	except Exception as e:
		# If an error occurs, log a message and return an empty dictionary
		COLLECT_AND_COMBINE_IMAGES_LOGGER.info(f"No classes are retrieved from directory.")
		return {}

def get_random_images(
		image_paths: list,
		num_samples: int,
		seed: int,
	) -> list:
	"""
	Retrieves a specified number of random image paths from a list of paths.

	Args:
		image_paths (list): A list of image paths.
		num_samples (int): The number of random images to retrieve. If None, the method will retrieve all image paths.
		seed (int): The seed used to generate the random sample.

	Returns:
		list: A list of image paths randomly selected from the input list.
	"""

	try:
		# Seed the random number generator for reproducibility
		random.seed(seed)

		# Retrieve the minimum of the number of samples requested and the number of paths available
		# This prevents trying to retrieve more samples than are available
		num_samples_to_retrieve = min(len(image_paths) if num_samples is None else num_samples, len(image_paths))

		# Return a list of random image paths
		return random.sample(image_paths, num_samples_to_retrieve)

	except Exception as e:
		# If an error occurs, log a message and re-raise the exception
		raise SIBIClassificationException(e, sys)


def collect_and_combine_images(
		classes: list=[], 
		train_path: str=None, 
		valid_path: str=None,
		test_path: str=None, 
		pattern_regex: str=r"\.(jpe?g)$", 
		num_images_per_class: dict=None, 
		seed: int=42
	) -> list:
	"""
	Collects all images from the given directories and combines them 
	into a single list. The list is then filtered to only include the
	given number of images per class.

	Args:
		classes (list): List of classes to collect images for.
		train_path (str): Path to the training data directory.
		valid_path (str): Path to the validation data directory.
		test_path (str): Path to the test data directory.
		pattern_regex (str): Pattern to match file extensions.
		num_images_per_class (dict): Dictionary of classes and number of images per class.
		seed (int): The seed used to generate the random sample.

	Returns:
		list: A list of image paths.
	"""

	try:
		# Define a helper function to process each class
		def process_class(cls):
			"""
			Combines images from training and validation for each class
			and filters the list to only include the given number of images
			per class.
			"""
			# Combine images from training and validation for each class
			all_train_images = train_images_paths.get(cls, [])
			all_valid_images = valid_images_paths.get(cls, [])
			all_test_images = test_images_paths.get(cls, [])
			all_combined_images = all_train_images + all_valid_images + all_test_images

			# Filter the list to only include the given number of images per class
			return get_random_images(
				image_paths=all_combined_images,
				num_samples=None if num_images_per_class is None else num_images_per_class.get(cls, len(all_combined_images)),
				seed=seed
			)

		# Print a title for the collection step
		custom_title_print(f"COLLECT {classes} FROM TRAINING DATA")

		# Collect images from the training data directory
		train_images_paths = collect_images_with_regex_and_count(train_path, classes, pattern_regex)
		custom_title_print(f"=")

		# Print a title for the validation step
		custom_title_print(f"COLLECT {classes} FROM VALIDATION DATA")

		# Collect images from the validation data directory
		valid_images_paths = collect_images_with_regex_and_count(valid_path, classes, pattern_regex)
		custom_title_print(f"=")

		# Print a title for the test step
		custom_title_print(f"COLLECT {classes} FROM TEST DATA")

		# Collect images from the test data directory
		test_images_paths = collect_images_with_regex_and_count(test_path, classes, pattern_regex)
		custom_title_print(f"=")

		# Print a title for the combination step
		custom_title_print(f"COMBINING {classes} FROM TRAINING AND VALIDATION DATA")

		# Process each class and filter the list to only include the given number of images per class
		random_images = {}

		# Use a ThreadPoolExecutor to process each class in parallel
		with ThreadPoolExecutor() as executor:
			results = executor.map(process_class, classes)

		# Combine the results into a single dictionary
		for cls, images in zip(classes, results):
			random_images[cls] = images
			COLLECT_AND_COMBINE_IMAGES_LOGGER.info(f"Total {log_manager.color_text(cls, COLOR_TEXT)} taken: {log_manager.color_text(len(random_images[cls]), COLOR_TEXT)}")

		# Combine all images into a single list
		all_images_paths = sum(random_images.values(), [])
		all_images_paths = [str(path) for path in all_images_paths]

		# Print a title for the final result
		custom_title_print(f"Total images taken: {len(all_images_paths)}")

		# Return the combined list of image paths
		return all_images_paths

	except Exception as e:
		# If an error occurs, log a message and re-raise the exception
		raise SIBIClassificationException(e, sys)
