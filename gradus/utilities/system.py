"""# grardus.utilities.system

Utility functions for system/hardware information and management.
"""

__all__ =   [
                "determine_device",
                "get_system_core_count",
            ]

from os     import cpu_count
from typing import Literal, Union

from torch  import cuda, device as t_device


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
    if device == "cpu":     return device("cpu")

    # Otherwise, if CUDA is available...
    if cuda.is_available(): return device("cuda")

    # If CUDA, is not available, we're using CPU.
    return device("cpu")


def get_system_core_count() -> int:
    """# Get System Core Count

    ## Returns:
        * (int):  Number of available CPU cores.
    """
    # Count cores.
    try:                return cpu_count() or 1

    # Should any complications arise, default to 1.
    except Exception:   return 1