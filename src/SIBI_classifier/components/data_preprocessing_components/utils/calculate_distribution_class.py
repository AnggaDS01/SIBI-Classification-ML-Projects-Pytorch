import sys
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torch.utils.data import Dataset
from SIBI_classifier.exception import SIBIClassificationException
from SIBI_classifier.logger.logging import *

COLOR_TEXT = 'yellow'

def calculate_class_distribution_torch(
        dataset: Dataset=None,  # The dataset from which we want to calculate the class distribution. This dataset should be a torch.utils.data.Dataset.
        class_labels: list=None,  # The list of class labels. This list should contain all possible class labels.
        class_weights_cvt_to_dict: bool=True # Whether to convert the class weights to a dictionary
    ) -> tuple:
    """
    This function calculates the class distribution and the corresponding weights for each class in the given dataset.
    
    The class distribution is calculated by counting the number of occurrences of each class label in the dataset.
    The class weights are calculated by using the compute_class_weight function from sklearn.utils.class_weight.
    
    Args:
        dataset (Dataset): The dataset from which we want to calculate the class distribution. This dataset should be a torch.utils.data.Dataset.
        class_labels (list): The list of class labels. This list should contain all possible class labels.
    
    Returns:
        tuple: A tuple containing the class distribution and the class weights. The class distribution is a dictionary where the keys are the class labels and the values are the number of occurrences of each class label. The class weights is a dictionary where the keys are the class labels and the values are the weights for each class.
    """

    try:
        # Get the class names from the dataset
        class_names = [data[1] for data in dataset]
        
        # Count the number of occurrences of each class label
        class_counts = Counter(class_names)
        
        # Get the class indices by finding the index of each class label in the class_labels list
        class_indices = [class_labels.index(name) for name in class_names]
        
        # Calculate the class weights using the compute_class_weight function from sklearn.utils.class_weight
        class_weight_values = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_indices),
            y=class_indices
        )
        print(class_weight_values)
        # Create a dictionary where the keys are the class labels and the values are the weights for each class
        if class_weights_cvt_to_dict:
            class_weights = {i: weight for i, weight in enumerate(class_weight_values)}
        else:
            class_weights = class_weight_values

        # Return the class distribution and the class weights
        return class_counts, class_weights

    except Exception as e:
        # If an error occurs, raise a SIBIClassificationException
        raise SIBIClassificationException(e, sys)
    
def print_class_distribution(
        distribution: Counter=None
    ) -> None:
    """
    This function prints the class distribution. The class distribution is a Counter object where the keys are the class labels and the values are the number of occurrences of each class label.

    Args:
        distribution (Counter): The Counter object containing the class distribution.
    """
    try:
        # Iterate over the class distribution and print the class name and the number of occurrences of each class label
        # The sorted function is used to sort the items in the Counter object so that the class labels are printed in alphabetical order
        for class_name, count in sorted(distribution.items()):
            CLASS_DISTRIBUTION_LOGGER.info(
                f"class {log_manager.color_text(class_name, COLOR_TEXT)}: {log_manager.color_text(count, COLOR_TEXT)} items"
            )

    except Exception as e:
        # If an error occurs, raise a SIBIClassificationException
        raise SIBIClassificationException(e, sys)
