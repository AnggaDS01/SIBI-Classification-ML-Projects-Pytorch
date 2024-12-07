import sys
import torch
import numpy as np
import random

from SIBI_classifier.exception import SIBIClassificationException


def set_seed(seed):
    """
    Sets the seeds for various random number generators used in this application.

    The seed is used to ensure reproducibility of the results. The same seed will produce the same results.

    Args:
        seed: The seed to use.
    
    Returns: 
        None
    """
    try:
        # Set the seed for the random module
        random.seed(seed)

        # Set the seed for numpy
        np.random.seed(seed)

        # Set the seed for torch
        torch.manual_seed(seed)

        # Set the seed for torch's CUDA backend if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set cudnn to be deterministic (i.e. to produce the same results every time)
        torch.backends.cudnn.deterministic = True

        # Disable cudnn's benchmarking mode (which can produce different results every time)
        torch.backends.cudnn.benchmark = False

    except Exception as e:
        raise SIBIClassificationException(e, sys)
