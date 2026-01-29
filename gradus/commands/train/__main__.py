"""# gradus.commands.train.main

Main process entry point for train command.
"""

__all__ = ["train_entry_point"]

from typing                             import Any, Dict, Literal, Union

from torch                              import device as t_device

from gradus.commands.train.__args__     import TrainConfig
from gradus.commands.train.utilities    import compute_accuracy
from gradus.registration                import register_command

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
                                            # max_workers =   get_system_core_count(),
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
                                            lr =            0.01
                                        )


    # For each epoch prescribed...
    for epoch in range(1, epochs + 1):

        # Place network in training mode.
        network.train()

        with tqdm(
            total = len(dataset.train_loader) + len(dataset.test_loader),
            desc =  f"Epoch {epoch}/{epochs}",
            unit =  "batch",
            leave = False
        ) as progress_bar:
            
            # Set progress status.
            progress_bar.set_postfix(status = f"Training")

            # For each batch in the training dataset...
            for samples, targets in dataset.train_loader:

                # Place samples and targets on device.
                samples, targets =          samples.to(device), targets.to(device)

                # Forward pass.
                predictions:    Tensor =    network(samples)

                # Compute accuracy.
                train_accuracy: float =     compute_accuracy(
                                                predictions =   predictions,
                                                targets =       targets
                                            )

                # Compute loss.
                train_loss:     Tensor =    cross_entropy(input = predictions, target = targets)

                # Reset weights.
                optimizer.zero_grad()

                # Backward pass & optimization step.
                train_loss.backward()

                # Update weights.
                optimizer.step()

                # Update progress bar.
                progress_bar.update(1)

            # Place network in evaluation mode.
            network.eval()
                
            # Set progress status.
            progress_bar.set_postfix(status = f"Validating")

            # For each batch in the training dataset...
            for samples, targets in dataset.test_loader:

                # Place samples and targets on device.
                samples, targets =          samples.to(device), targets.to(device)

                # Forward pass.
                predictions:    Tensor =    network(samples)

                # Compute accuracy.
                val_accuracy: float =       compute_accuracy(
                                                predictions =   predictions,
                                                targets =       targets
                                            )

                # Compute loss.
                val_loss:       Tensor =    cross_entropy(input = predictions, target = targets)

                # Update progress bar.
                progress_bar.update(1)

        # Log epoch results.
        logger.info(
            f"Epoch {epoch}/{epochs}: "
            f"Train Accuracy = {train_accuracy:.4f}; "
            f"Train Loss = {train_loss:.4f}; "
            f"Validation Accuracy = {val_accuracy:.4f}; "
            f"Validation Loss = {val_loss:.4f}"
        )