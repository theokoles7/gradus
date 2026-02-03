"""# gradus.commands.train.main

Main process entry point for train command.
"""

__all__ = ["train_entry_point"]

from typing                             import Any, Dict, Literal, Union

from torch                              import device as t_device

from gradus.commands.train.__args__     import TrainConfig
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
        * dataset_id    (str):          Identifier of dataset upon which neural network will be 
                                        trained.
        * epochs        (int):          Number of training/validation epochs to administer. Defaults 
                                        to 100.
        * output_path   (str):          Path at which training results will be written. Defaults to 
                                        "results".
        * seed          (int):          Random seed for reproducibility. Defaults to 1.
        * device        (str | device): Hardware device upon which data will be processed. Defaults 
                                        to "auto".

    ## Returns:
        * Dict[str, Any]:   Training results.
    """
    from logging                import Logger

    from torch                  import no_grad, Tensor
    from torch.nn               import Module
    from torch.nn.functional    import cross_entropy
    from torch.optim            import SGD
    from tqdm                   import tqdm

    from gradus.artifacts       import TrainingRecord
    from gradus.datasets        import Dataset
    from gradus.registration    import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities       import determine_device, get_logger, set_seed

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
                                            **kwargs
                                        )

    # Load neural network.
    network:        Module =            NETWORK_REGISTRY.load_network(
                                            network_id =    network_id,
                                            input_shape =   dataset.input_shape,
                                            num_classes =   dataset.num_classes,
                                            **kwargs
                                        ).to(device)

    # Initialize optimizer.
    optimizer:      SGD =               SGD(params = network.parameters(), lr = 0.01)
    
    # Initialize training data map.
    train_record:   TrainingRecord =    TrainingRecord(
                                            output_path =   f"{output_path}/{network_id}_{dataset_id}"
                                        )

    # Log action.
    logger.info(f"Train process initiated (network = {network_id}; dataset = {dataset_id})")


    # For each epoch prescribed...
    for epoch in range(1, epochs + 1):

        with tqdm(
            total = len(dataset.train_loader) + len(dataset.test_loader),
            desc =  f"Epoch {epoch}/{epochs}",
            unit =  "batch",
            leave = False
        ) as progress_bar:

            # Initialize metric trackers.
            train_total_loss, train_correct, train_total =  0, 0, 0
            val_total_loss,   val_correct,   val_total =    0, 0, 0

            # Place network in training mode.
            network.train(); progress_bar.set_postfix(status = f"Training")

            # For each batch in the training dataset...
            for samples, targets in dataset.train_loader:

                # Place samples and targets on device.
                samples, targets =          samples.to(device), targets.to(device)

                # Forward pass.
                predictions:    Tensor =    network(samples)

                # Compute loss.
                loss:           Tensor =    cross_entropy(input = predictions, target = targets)

                # Back propagation.
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                # Update metric trackers.
                train_total_loss +=         loss.item()
                _, predictions =            predictions.max(dim = 1)
                train_correct +=            predictions.eq(targets).sum().item()
                train_total +=              targets.size(0)

                # Update progress bar.
                progress_bar.update(1)

            # Place network in evaluation mode.
            network.eval(); progress_bar.set_postfix(status = f"Validating")

            # For each batch in the test dataset...
            for samples, targets in dataset.test_loader:

                # Place samples and targets on device.
                samples, targets =          samples.to(device), targets.to(device)

                # Forward pass.
                with no_grad(): predictions: Tensor = network(samples)

                # Compute loss.
                loss:           Tensor =    cross_entropy(input = predictions, target = targets)

                # Update metric trackers.
                val_total_loss +=           loss.item()
                _, predictions =            predictions.max(dim = 1)
                val_correct +=              predictions.eq(targets).sum().item()
                val_total +=                targets.size(0)

                # Update progress bar.
                progress_bar.update(1)

        # Compute average accuracy & loss.
        train_accuracy: float = round(train_correct / train_total,                  4)
        train_loss:     float = round(train_total_loss / len(dataset.train_loader), 4)
        val_accuracy:   float = round(val_correct / val_total,                      4)
        val_loss:       float = round(val_total_loss / len(dataset.test_loader),    4)
            
        # Record epoch results.
        train_record.record_epoch(epoch, train_accuracy, train_loss, val_accuracy, val_loss)

        # Log epoch results.
        logger.info(
            f"Epoch {epoch}/{epochs}: "
            f"Train Accuracy = {train_accuracy:.4f}; "
            f"Train Loss = {train_loss:.4f}; "
            f"Validation Accuracy = {val_accuracy:.4f}; "
            f"Validation Loss = {val_loss:.4f}"
        )