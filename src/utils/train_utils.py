import random

import numpy as np
import torch


def set_seed(seed):
    """
    Set the seed for reproducibility in Python's random, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to set.
    """
    random.seed(seed)  # Set seed for Python's standard random library
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch CPU operations

    if torch.cuda.is_available():
        # Set seed for PyTorch CUDA operations
        torch.cuda.manual_seed(seed)
        # Set seed for all GPUs (if using more than one)
        torch.cuda.manual_seed_all(seed)
        # Ensure CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        # Disable dynamic algorithm selection for convolution
        torch.backends.cudnn.benchmark = False
