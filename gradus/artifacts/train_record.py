"""# gradus.artifacts.train_results

Training results map structure & utility.
"""

__all__ = ["TrainingRecord"]

from logging            import Logger
from pathlib            import Path
from typing             import Any, Dict, List, Union

from gradus.utilities   import get_logger

class TrainingRecord():
    """# Training Data Record Keeping"""

    def __init__(self,
        output_path:    Union[str, Path] =  "results"
    ):
        """# Instantiate Training Record.

        ## Args:
            * output_path   (str | Path):   Path at which training results will be written. 
                                            Defaults to "results".
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger("train-record")

        # Define properties.
        self._output_path_: Path =              Path(output_path)
        self._epochs_:      Dict[int, Dict] =   {}

        # Ensure output path exists.
        self._output_path_.mkdir(parents = True, exist_ok = True)

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def best_epoch(self) -> int:
        """# Epoch with Best Validation Accuracy"""
        return  max(
                    self._epochs_,
                    key =   lambda epoch: self._epochs_[epoch]["validation"]["accuracy"]
                )

    @property
    def num_epochs(self) -> int:
        """# Quantity of Epochs Recorded"""
        return len(self._epochs_)

    @property
    def train_accuracies(self) -> List[float]:
        """# Train Accuracy Sequence"""
        return [e["train"]["accuracy"] for e in self._epochs_.values()]
    
    @property
    def train_losses(self) -> List[float]:
        """# Train Loss Sequence"""
        return [e["train"]["loss"] for e in self._epochs_.values()]
    
    @property
    def validation_accuracies(self) -> List[float]:
        """# Validation Accuracy Sequence"""
        return [e["validation"]["accuracy"] for e in self._epochs_.values()]
    
    @property
    def validation_losses(self) -> List[float]:
        """# Validation Loss Sequence"""
        return [e["validation"]["loss"] for e in self._epochs_.values()]

    # METHODS ======================================================================================

    def record_epoch(self,
        epoch:          int,
        train_accuracy: float,
        train_loss:     float,
        val_accuracy:   float,
        val_loss:       float
    ) -> None:
        """# Record Epoch Results.

        ## Args:
            * accuracy  (float):    Model's classification accuracy.
            * loss      (float):    Classification loss score.
        """
        # Record epoch data.
        self._epochs_[epoch] =  {
                                    "train":        {
                                                        "accuracy": train_accuracy,
                                                        "loss":     train_loss
                                                    },
                                    "validation":   {
                                                        "accuracy": val_accuracy,
                                                        "loss":     val_loss
                                                    }
                                }
        
        # Debug record.
        self.__logger__.debug(f"Recorded Epoch {epoch}: {self._epochs_[epoch]}")

    def save_to_master(self,
        network_id: str,
        dataset_id: str,
        device:     str,
        seed:       int,
        epochs:     int,
        accuracy:   float,
        loss:       float,
        metric:     str =   None,
        sampler:    str =   None,
        order:      str =   None
    ) -> None:
        """# Save to Master Record.

        ## Args:
            * network_id    (str):      Network identifier.
            * dataset_id    (str):      Dataset identifier.
            * device        (str):      Device used for run.
            * seed          (int):      Random seed used for run.
            * epochs        (int):      Number of epochs trained.
            * accuracy      (float):    Final validation accuracy.
            * loss          (float):    Final validation loss.
            * metric        (str):      Curriculum metric ID, if any.
            * sampler       (str):      Curriculum sampler ID, if any.
            * order         (str):      Curriculum ordering ID, if any.
        """
        from csv    import DictWriter
        
        # Define CSV fields.
        FIELDS:         List[str] = [
                                        "network_id", "dataset_id", "device", "seed", "epochs",
                                        "accuracy", "loss", "metric", "sampler", "order"
                                    ]
        
        # Resolve results CSV path.
        results_path:   Path =      Path("results/master_record.csv")

        # Ensure path exists.
        results_path.parent.mkdir(parents = True, exist_ok = True)

        # If file does not exist or is empty...
        if not results_path.exists() or results_path.stat().st_size == 0:

            # Open file for writing.
            with open(results_path, "w", newline = "") as f:

                # Write header.
                DictWriter(f, fieldnames = FIELDS).writeheader()

        # Open file for writing.
        with open(results_path, "a", newline = "") as f:

            # Write record.
            DictWriter(f, fieldnames = FIELDS).writerow({
                "network_id":   network_id,
                "dataset_id":   dataset_id,
                "seed":         seed,
                "device":       str(device),
                "epochs":       epochs,
                "metric":       metric,
                "sampler":      sampler,
                "order":        order,
                "accuracy":     accuracy,
                "loss":         loss,
            })

        self.__logger__.info(f"Result saved to master record at {results_path.absolute}")

    def to_dict(self) -> Dict[str, Any]:
        """# Dictionary Representation of Training Record.

        ## Returns:
            * Dict[str, Any]:   Training data/results.
        """
        # Notate epoch quantity only once.
        epoch_qty:  int =   self.num_epochs

        # Provide mapping of training data/results.
        return  {
                    "epochs":               self._epochs_,
                    "best_epoch":           self.best_epoch,
                    "avg_train_accuracy":   sum(self.train_accuracies)      / epoch_qty,
                    "avg_train_loss":       sum(self.train_losses)          / epoch_qty,
                    "avg_val_accuracy":     sum(self.validation_accuracies) / epoch_qty,
                    "avg_val_losses":       sum(self.validation_losses)     / epoch_qty
                }
        
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Training Record Object Representation"""
        return f"""<TrainingRecord({self.num_epochs} epoch(s))>"""