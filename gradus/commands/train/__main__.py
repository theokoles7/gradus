"""# gradus.commands.train.main

Main process entry point for train command.
"""

__all__ = ["train_entry_point"]

from pathlib                            import Path
from typing                             import Any, Dict, List, Literal, Union

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
    epochs:         int =                                               200,
    seed:           int =                                               1,
    device:         Union[Literal["auto", "cpu", "cuda"], t_device] =   "auto",
    output_path:    str =                                               "results",
    cache_path:     Union[str, Path] =                                  ".cache",
    *args,
    **kwargs
) -> Dict[str, Any]:
    """# Train Neural Network on Dataset.

    ## Args:
        * network_id    (str):          Identifier of neural network being trained.
        * dataset_id    (str):          Identifier of dataset upon which neural network will be 
                                        trained.
        * epochs        (int):          Number of training/validation epochs to administer. Defaults 
                                        to 200.
        * seed          (int):          Random seed for reproducibility. Defaults to 1.
        * device        (str | device): Hardware device upon which data will be processed. Defaults 
                                        to "auto".
        * output_path   (str):          Path at which training results will be written. Defaults to 
                                        "./results/".
        * cache_path    (str | Path):   Path at which training artifacts will be cached. Defaults to 
                                        "./.cache/".

    ## Returns:
        * Dict[str, Any]:   Training results.
    """
    from logging                    import Logger
    from pathlib                    import Path

    from torch                      import no_grad, Tensor
    from torch.nn.functional        import cross_entropy
    from torch.optim                import SGD
    from torch.optim.lr_scheduler   import CosineAnnealingLR
    from tqdm                       import tqdm

    from gradus.artifacts           import TrainingRecord
    from gradus.datasets            import Dataset
    from gradus.networks            import Network
    from gradus.registration        import DATASET_REGISTRY, NETWORK_REGISTRY, SCHEDULE_REGISTRY
    from gradus.utilities           import determine_device, get_logger, set_seed

    # Initialize logger.
    logger:         Logger =            get_logger("train-process")

    # Set seed.
    set_seed(seed = seed);              logger.info(f"Seed: {seed}")

    # Determine device.
    device:          t_device =           determine_device(
                                            device = device
                                        ); logger.info(f"Device: {device}")

    # Load dataset.
    dataset:        Dataset =           DATASET_REGISTRY.load_dataset(
                                            dataset_id =    dataset_id,
                                            seed =          seed,
                                            epochs =        epochs,
                                            **kwargs
                                        )

    # Load neural network.
    network:        Network =           NETWORK_REGISTRY.load_network(
                                            network_id =    network_id,
                                            input_shape =   dataset.input_shape,
                                            num_classes =   dataset.num_classes,
                                            **kwargs
                                        ).to(device)
    
    # Determine if this training is using adaptive scheduling.
    adaptive:       bool =              (
                                            dataset.schedule is not None and
                                            dataset.schedule.id in  SCHEDULE_REGISTRY.list_entries(
                                                                        filter_by = ["adaptive"]
                                                                    )
                                            
                                        )

    # Initialize optimizer.
    optimizer:      SGD =               SGD(
                                            params =        network.parameters(),
                                            lr =            0.1,
                                            weight_decay =  5e-4,
                                            momentum =      0.9
                                        )

    # Define learning rate annealing scheduler.
    scheduler:      CosineAnnealingLR = CosineAnnealingLR(
                                            optimizer =     optimizer,
                                            T_max =         epochs
                                            )
    
    # Initialize training data map.
    train_record:   TrainingRecord =    TrainingRecord(
                                            network_config =    network.dict,
                                            dataset_config =    dataset.dict,
                                            epochs =            epochs,
                                            device =            device,
                                            seed =              seed,
                                            output_path =       output_path,
                                            cache_path =        cache_path,
                                            max_batches =       len(dataset.train_loader)
                                        )
    
    # If training record already exists...
    if train_record.already_exists:

        # No need to do it again.
        logger.info(f"Training already recorded (hash = {train_record.hash})"); return

    # Log action.
    logger.info(f"Train process initiated (network = {network}; dataset = {dataset}' epochs = {epochs})")

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

            # Initialize per-batch signal accumulators.
            batch_std_rows:     List[Dict] =                []
            batch_grad_norm:    List[Dict] =                []

            # Place network in training mode.
            network.train(); progress_bar.set_postfix(status = f"Training")

            # Extract curriculum indices.
            curriculum_indices: List[int] = (
                                                dataset.curriculum.batch_indices
                                                if dataset.curriculum is not None
                                                else None
            )
            
            # print(f"len(curriculum_indices): {len(curriculum_indices) if curriculum_indices else None}")
            # print(f"len(dataset.train_loader): {len(dataset.train_loader)}")

            # For each batch in the training dataset...
            for b, (samples, targets) in enumerate(dataset.train_loader):

                # Place samples and targets on device.
                samples, targets =          samples.to(device), targets.to(device)

                # If using adaptive scheduling
                if adaptive:    predictions, activations =  network(samples, return_activations = True)
                else:           predictions: Tensor =       network(samples)

                # Compute loss.
                loss:           Tensor =    cross_entropy(input = predictions, target = targets)

                # Back propagation.
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                # Update metric trackers.
                train_total_loss +=         loss.item()
                _, predictions =            predictions.max(dim = 1)
                train_correct +=            predictions.eq(targets).sum().item()
                train_total +=              targets.size(0)

                # If adaptive scheduling is used...
                if adaptive:

                    # Compute mean activation standard deviation across all layers.
                    mean_std:       float =         sum(act.std().item() for act in activations)    \
                                                    / len(activations)
                    
                    # Compute gradient L2 norm across all layers wih gradients.
                    grad_norms:     List[float] =   [
                                                        p.grad.data.norm(2).item()
                                                        for p in network.parameters()
                                                        if p.grad is not None
                                                    ]
                    
                    # Compute mean of L2 norm.
                    mean_grad_norm: float =         sum(grad_norms) / len(grad_norms) \
                                                    if grad_norms else 0.0
                    
                    # Use original curriculum batch index, not loop counter.
                    actual_idx: int = curriculum_indices[b] if curriculum_indices is not None else b
                    
                    # Accumulate records.
                    batch_std_rows.append( {"batch_idx": b, "mean_std":       mean_std})
                    batch_grad_norm.append({"batch_idx": b, "mean_grad_norm": mean_grad_norm})

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
        train_record.record_epoch(
            epoch =             epoch,
            train_accuracy =    train_accuracy,
            train_loss =        train_loss,
            val_accuracy =      val_accuracy,
            val_loss =          val_loss,
            batches =           len(dataset.train_loader)
        )

        # Log epoch results.
        logger.info(
            f"Epoch {epoch:3}/{epochs}: "
            f"Train Accuracy = {train_accuracy:.4f}; "
            f"Train Loss = {train_loss:.4f}; "
            f"Validation Accuracy = {val_accuracy:.4f}; "
            f"Validation Loss = {val_loss:.4f}"
        )

        # Anneal learning rate schedule.
        scheduler.step()

        # Build per-batch signal DataFrames for adaptive schedule.
        if adaptive:
            from pandas import DataFrame

            std_df:         DataFrame = DataFrame(batch_std_rows).set_index("batch_idx")       \
                                        if batch_std_rows       else DataFrame()
            grad_norm_df:   DataFrame = DataFrame(batch_grad_norm).set_index("batch_idx") \
                                        if batch_grad_norm else DataFrame()
        else:
            std_df =        None
            grad_norm_df =  None

        # Advance curriculum pacing schedule.
        dataset.step(
            epoch =         epoch,
            loss =          train_loss,
            val_acc =       val_accuracy,
            std_df =        std_df,
            grad_norm_df =  grad_norm_df
        )

    # Save model weights.
    network.save_weights(path = train_record.weights_path)

    # Save training record.
    train_record.save()

    # Communicate weights location.
    logger.info(f"Weights saved to {train_record.weights_path}")