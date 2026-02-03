"""# grardus.utilities.system

Utility functions for system/hardware information and management.
"""

__all__ =   [
                "determine_device",
                "get_system_core_count",
                "set_seed",
            ]

from os                 import cpu_count
from random             import seed as r_seed
from typing             import Literal, Union

from numpy.random       import seed as np_seed
from torch              import cuda, device as t_device, manual_seed
from torch.backends     import cudnn


def determine_device(
    device: Union[t_device, Literal["auto", "cpu", "cuda"]]
) -> t_device:
    """# Determine Data Processing Device.

    ## Args:
        * device    (str |  device):    Intended device.

    ## Returns:
        * t_device: Best available device based on provided choice.
    """
    # If CPU is chosen, simply return CPU.
    if device == "cpu":     return t_device("cpu")

    # Otherwise, if CUDA is available...
    if cuda.is_available(): return t_device("cuda")

    # If CUDA, is not available, we're using CPU.
    return t_device("cpu")


def get_system_core_count() -> int:
    """# Get System Core Count.

    ## Returns:
        * (int):  Number of available CPU cores.
    """
    # Count cores.
    try:                return cpu_count() or 1

    # Should any complications arise, default to 1.
    except Exception:   return 1
    

def set_seed(
    seed:   int
) -> None:
    """# Set Random Number Generation Seed.

    ## Args:
        * seed  (int):  Random number generation seed.
    """
    # Set seeds.
    r_seed(seed)
    np_seed(seed)
    manual_seed(seed)

    # If CUDA is available...
    if cuda.is_available():

        # Configure deterministic computing.
        cuda.manual_seed(seed)
        cuda.manual_seed_all(seed)
        cudnn.deterministic =   True
        cudnn.benchmark =       False