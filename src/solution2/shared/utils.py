import torch
import os

def get_device() -> torch.device:
    """
    Returns the best available device: CUDA -> MPS -> CPU.
    Also prints the device to the console.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def set_seed(seed: int) -> None:
    """
    Ensures reproducibility across random, numpy, and torch.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: MPS reproducibility is still maturing; manual_seed covers basic cases.
