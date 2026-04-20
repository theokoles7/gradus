"""# gradus.artifacts.train_results

Training results map structure & utility.
"""

__all__ = ["TrainingRecord"]

from functools          import cached_property
from hashlib            import md5
from json               import dumps
from logging            import Logger
from pathlib            import Path
from typing             import Any, Dict, List, Optional, Union

from torch              import device as t_device

from gradus.utilities   import get_logger

class TrainingRecord():
    """# Training Data Record Keeping"""

    def __init__(self,
        network_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        epochs:         int,
        device:         t_device,
        seed:           int,
        output_path:    Union[str, Path] =  "results",
        cache_path:     Union[str, Path] =  ".cache",
        max_batches:    int =               0
    ):
        """# Instantiate Training Record.

        ## Args:
            * network_config    (Dict[str, Any]):   Network configuration map.
            * dataset_config    (Dict[str, Any]):   Dataset configuration map.
            * device            (str):              Device used for run.
            * seed              (int):              Random seed used for run.
            * epochs            (int):              Number of epochs trained.
            * output_path       (str | Path):       Path at which training results will be written. 
                                                    Defaults to "./results/".
            * cache_path        (str | Path):       Path at which training artifacts will be cached. 
                                                    Defaults to "./.cache/".
        """
        # Initialize logger.
        self.__logger__:        Logger =            get_logger("train-record")

        # Define properties.
        self._network_config_:  Dict[str, Any] =    network_config
        self._dataset_config_:  Dict[str, Any] =    dataset_config
        self._num_epochs_:      int =               epochs
        self._device_:          t_device =          device
        self._seed_:            int =               seed
        self._epochs_:          Dict[int, Dict] =   {}
        self._max_batches_:     int =               max_batches

        # Resolve paths.
        self._output_path_:     Path =              Path(output_path);  self._output_path_.mkdir(
                                                                            parents = True,
                                                                            exist_ok = True
                                                                        )
        self._cache_path_:      Path =              Path(cache_path);   self._cache_path_.mkdir(
                                                                            parents =   True,
                                                                            exist_ok =  True
                                                                        )
        

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def already_exists(self) -> bool:
        """# Training Process is Already Cached?"""
        # If master record does not exist, no trainings have been conducted.
        if not self.master_record_path.exists(): return False

        # Otherwise, import dictionary reader.
        from csv    import DictReader

        # Open master record for reading.
        with open(self.master_record_path, "r") as master_record:

            # Indicate if training process has already been recorded.
            return  any(
                        row.get("hash") == self.hash
                        for row in DictReader(master_record)
                    )
        
    @property
    def best_accuracy(self) -> float:
        """# Validation Accuracy at Best Epoch"""
        return self._epochs_[self.best_epoch]["validation"]["accuracy"]

    @property
    def best_epoch(self) -> int:
        """# Epoch with Best Validation Accuracy"""
        return  max(
                    self._epochs_,
                    key =   lambda epoch: self._epochs_[epoch]["validation"]["accuracy"]
                )
    
    @property
    def best_loss(self) -> float:
        """# Validation Loss at Best Epoch"""
        return self._epochs_[self.best_epoch]["validation"]["loss"]
    
    @property
    def config(self) -> Dict[str, Any]:
        """# Training Process Configuration Metadata"""
        return  {
                    "network":      self._network_config_,
                    "dataset":      self._dataset_config_,
                    "seed":         self._seed_,
                    "device":       str(self._device_),
                    "num_epochs":   self._num_epochs_,
                }
    
    @property
    def dsi(self) -> Optional[float]:
        """# Data Saturation Index

        Sum of the number of batches used at each epoch divided by the total number of batches in
        the dataset (one-epoch size). For a baseline run using the full dataset every epoch, this
        equals ``num_epochs``; curriculum runs that skip data score lower. Returns None when no
        batch counts or max-batches reference is available.
        """
        # Extract recorded batch counts.
        batch_counts:   List[int] = [e["batches"] for e in self._epochs_.values()]

        # If no batch counts were recorded, DSI is not applicable.
        if not any(b is not None for b in batch_counts): return None

        # If per-epoch max-batches reference is unavailable, DSI is not applicable.
        if not self._max_batches_: return None

        # Sum the number of batches processed across all epochs.
        processed:      int =       sum(b for b in batch_counts if b is not None)

        # Compute DSI: fraction of dataset processed, summed across epochs.
        return round(processed / self._max_batches_, 4)

    @property
    def batches_per_epoch(self) -> List[Optional[int]]:
        """# Batches Processed at Each Recorded Epoch"""
        return [e["batches"] for e in self._epochs_.values()]

    @property
    def max_batches(self) -> int:
        """# Batches Available Per Epoch (Full Dataset Size)"""
        return self._max_batches_

    @property
    def total_batches_processed(self) -> int:
        """# Total Batches Processed Across All Epochs"""
        return sum(b for b in self.batches_per_epoch if b is not None)
    
    @property
    def final_accuracy(self) -> float:
        """# Validation Accuracy at Last Recorded Epoch"""
        return self.validation_accuracies[-1]
    
    @property
    def final_loss(self) -> float:
        """# Validation Loss at Last Recorded Epoch"""
        return self.validation_losses[-1]
    
    @cached_property
    def hash(self) -> str:
        """# Unique Hash of Training Variables"""
        return  md5(dumps({
                    "network":  self._network_config_,
                    "dataset":  self._dataset_config_,
                    "epochs":   self._num_epochs_,
                    "device":   str(self._device_),
                    "seed":     self._seed_
                }, sort_keys = True).encode()).hexdigest()
    
    @property
    def master_record_path(self) -> Path:
        """# Master Record File Path"""
        return self._output_path_ / "master_record.csv"

    @property
    def num_epochs(self) -> int:
        """# Quantity of Epochs Recorded"""
        return len(self._epochs_)
    
    @property
    def record_path(self) -> Path:
        """# Path at Which Training Record is Located"""
        return  (
                    self._output_path_                                                  /
                    f"""{self._network_config_["id"]}_{self._dataset_config_["id"]}"""  /
                    f"{self.hash}.json"
                )
    
    @property
    def results(self) -> Dict[str, Any]:
        """# Current Training Results"""
        return  {
            "final_accuracy":   self.final_accuracy,
            "final_loss":       self.final_loss,
            "best_accuracy":    self.best_accuracy,
            "best_loss":        self.best_loss,
            "best_epoch":       self.best_epoch
        }

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
    
    @property
    def weights_path(self) -> Path:
        """# Path at Which Network Weights Will be Stored"""
        return self._cache_path_ / "weights" / f"{self.hash}.pth"

    # METHODS ======================================================================================

    def record_epoch(self,
        epoch:          int,
        train_accuracy: float,
        train_loss:     float,
        val_accuracy:   float,
        val_loss:       float,
        batches:        int
    ) -> None:
        """# Record Epoch Results.

        ## Args:
            * epoch     (int):  Epoch being recorded.
            * train_accuracy    (float):    Train accuracy during epoch.
            * train_loss        (float):    Train loss during epoch.
            * val_accuracy      (float):    Validation accuracy during epoch.
            * val_loss          (float):    Validation loss during epoch.
            * batches           (int):      Number of batches used in epoch.
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
                                                    },
                                    "batches":      batches
                                }
        
        # Debug record.
        self.__logger__.debug(f"Recorded Epoch {epoch}: {self._epochs_[epoch]}")

    def save(self) -> None:
        """# Save Training Record."""
        # Save to master record.
        self._save_to_master_record_()

        # Save verbose record.
        self._save_verbose_record_()
        
    def to_dict(self) -> Dict[str, Any]:
        """# Dictionary Representation of Training Record.

        ## Returns:
            * Dict[str, Any]:   Training data/results.
        """
        # Notate epoch quantity only once.
        epoch_qty:  int =   self.num_epochs

        # Provide mapping of training data/results.
        return  {
                    "best_epoch":           self.best_epoch,
                    "best_val_accuracy":    self._epochs_[self.best_epoch]["validation"]["accuracy"],
                    "final_train_accuracy": self.train_accuracies[-1],
                    "final_train_loss":     self.train_losses[-1],
                    "final_val_accuracy":   self.validation_accuracies[-1],
                    "final_val_loss":       self.validation_losses[-1],
                    "dsi":                      self.dsi,
                    "max_batches_per_epoch":    self._max_batches_,
                    "total_batches_processed":  self.total_batches_processed,
                    "batches_per_epoch":        self.batches_per_epoch,
                    "epochs":               self._epochs_,
                }
    
    # HELPERS ======================================================================================

    def _save_to_master_record_(self) -> None:
        """# Save Training Results to Master Record."""
        from csv    import DictWriter

        # Define CSV fields.
        FIELDS: List[str] = [
                                "network_id", "dataset_id", "epochs", "seed", "device",
                                "final_accuracy", "final_loss", "best_accuracy", "best_loss",
                                "best_epoch", "shuffled", "normalize_classes", "rank", "metric",
                                "scope", "schedule", "start_fraction", "dsi",
                                "max_batches_per_epoch", "total_batches_processed",
                                "record_file", "hash"
                            ]
        
        # If master record does not exist, or is empty...
        if not self.master_record_path.exists() or self.master_record_path.stat().st_size == 0:
    
            # Open file for writing.
            with open(self.master_record_path, "w", newline = "") as master_record:

                # Write header.
                DictWriter(master_record, fieldnames = FIELDS).writeheader()
    
        # Open file for writing.
        with open(self.master_record_path, "a", newline = "") as master_record:

            # Reconcile curriculum, in case it is None.
            curriculum: Dict[str, Any] =    self._dataset_config_.get("curriculum") or {}

            # Write results to master record.
            DictWriter(master_record, fieldnames = FIELDS).writerow({
                "network_id":           self._network_config_["id"],
                "dataset_id":           self._dataset_config_["id"],
                "epochs":               self._num_epochs_,
                "seed":                 self._seed_,
                "device":               self._device_,
                "final_accuracy":       self.final_accuracy,
                "final_loss":           self.final_loss,
                "best_accuracy":        self.best_accuracy,
                "best_loss":            self.best_loss,
                "best_epoch":           self.best_epoch,
                "shuffled":             self._dataset_config_["shuffled"],
                "normalize_classes":    self._dataset_config_["normalize_classes"],
                "rank":                 curriculum.get("rank"),
                "metric":               curriculum.get("metric"),
                "scope":                curriculum.get("scope"),
                "schedule":             self._dataset_config_.get("schedule_id"),
                "start_fraction":       self._dataset_config_.get("start_fraction"),
                "dsi":                  self.dsi,
                "max_batches_per_epoch":    self._max_batches_,
                "total_batches_processed":  self.total_batches_processed,
                "record_file":          self.record_path,
                "hash":                 self.hash
            })

        # Communicate master record.
        self.__logger__.info(
            f"Result saved to master record at {self.master_record_path.absolute()}"
        )

    def _save_verbose_record_(self) -> None:
        """# Save Verbose Training Record."""
        from json   import dump

        # Ensure path to verbose record exists.
        self.record_path.parent.mkdir(parents = True, exist_ok = True)

        # Open verbose record for writing.
        with open(self.record_path, "w") as training_record:

            # Save verbose training record.
            dump({
                "config":   self.config,
                "results":  self.results,
                "dsi":                      self.dsi,
                "max_batches_per_epoch":    self._max_batches_,
                "total_batches_processed":  self.total_batches_processed,
                "batches_per_epoch":        self.batches_per_epoch,
                "epochs":   self._epochs_
            }, training_record, indent = 2, default = str)

        # Communicate verbose record path.
        self.__logger__.info(f"""Verbose record saved to {self.record_path.absolute()}""")
        
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Training Record Object Representation"""
        return f"""<TrainingRecord({self.num_epochs} epoch(s))>"""