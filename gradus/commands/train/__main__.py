"""# gradus.commands.train.main

Main process entry point for train command.
"""

__all__ = ["train_entry_point"]

from typing                         import Any, Dict, Union

from torch                          import device as t_device

from gradus.commands.train.__args__ import TrainConfig
from gradus.registration            import register_command

@register_command(
    id =        "train",
    config =    TrainConfig
)
def train_entry_point(
    network_id:     str,
    dataset_id:     str,
    epochs:         int =                   100,
    output_path:    str =                   "results",
    seed:           int =                   1,
    device:         Union[str, t_device] =  "auto",
    *args,
    **kwargs
) -> Dict[str, Any]:
    """# Train Neural Network on Dataset.

    ## Args:
        * network_id    (str):          Identifier of neural network being trained.
        * dataset_id    (str):          Identifier of dataset upon which neural network will be trained.
        * epochs        (int):          Number of training/validation epochs to administer. Defaults to 100.
        * output_path   (str):          Path at which training results will be written. Defaults to "results".
        * seed          (int):          Random seed for reproducibility. Defaults to 1.
        * device        (str | device): Hardware device upon which data will be processed. Defaults to "auto".

    ## Returns:
        * Dict[str, Any]:   Training results.
    """