"""# gradus.commands.train.main

Main process entry point for train command.
"""

__all__ = ["train_entry_point"]

from typing                         import Any, Dict, Literal, Union

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
    epochs:         int =                                               100,
    output_path:    str =                                               "results",
    seed:           int =                                               1,
    device:         Union[Literal["auto", "cpu", "cuda"], t_device] =   "auto",
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
    from logging                import Logger

    from torch                  import Tensor
    from torch.nn               import Module
    from torch.nn.functional    import cross_entropy
    from torch.optim            import Adam, SGD
    from tqdm                   import tqdm

    from gradus.artifacts       import TrainingRecord
    from gradus.datasets        import Dataset
    from gradus.registration    import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities       import determine_device, get_logger, get_system_core_count, set_seed

    # Initialize logger.
    logger:         Logger =            get_logger("train-process")

    # Set seed.
    set_seed(seed = seed);              logger.info(f"Seed: {seed}")

    # Determine device.
    device:         t_device =           determine_device(
                                            device = device
                                        ); logger.info(f"Device: {device}")

    # Load dataset.
    dataset:        Dataset =           DATASET_REGISTRY.load_dataset(
                                            dataset_id =    dataset_id,
                                            max_workers =   get_system_core_count(),
                                            **kwargs
                                        )

    # Load neural network.
    network:        Module =            NETWORK_REGISTRY.load_network(
                                            network_id =    network_id,
                                            input_shape =   dataset.input_shape,
                                            num_classes =   dataset.num_classes,
                                            **kwargs
                                        ).to(device)
    
    # Initialize training data map.
    train_record:   TrainingRecord =    TrainingRecord(output_path = output_path)

    # Log action.
    logger.info(f"Train process initiated (network = {network_id}; dataset = {dataset_id})")

    # Initialize optimizer.
    optimizer:      SGD =               SGD(
                                            params =        network.parameters(),

    # For each epoch prescribed...
    for epoch in tqdm(
        range(1, epochs + 1),
        desc =  "Training",
        unit =  "epoch",
        leave = True
    ): 
        # Place network in training mode.
        network.train()

        # For each batch in the training dataset...
        for samples, targets in dataset.train_loader:

            # Place samples and targets on device.
            samples, targets =          samples.to(device), targets.to(device)

            # Forward pass.
            predictions:    Tensor =    network(samples)

            # Compute loss.
            loss:           Tensor =    cross_entropy(input = predictions, target = targets)

            # Backward pass & optimization step.
            loss.backward()

            # 