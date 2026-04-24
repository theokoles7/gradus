"""# gradus.curricula.schedules.adaptive

Adaptive curriculum pacing schedule implementation.
"""

__all__ = ["AdaptiveSchedule"]

from math                                   import ceil
from typing                                 import Any, Dict, List, override

from gradus.curricula.schedules.protocol    import Schedule
from gradus.registration                    import register_schedule

@register_schedule(
    id =    "adaptive",
    tags =  ["pacing", "adaptive"]
)
class AdaptiveSchedule(Schedule):
    """# Adaptive Curriculum Pacing Schedule

    Controls the rate at which new curriculum data is introduced by monitoring four training signals
    - loss plateau, validation accuracy trend, activation stability, and gradient norm stability -
    and computing a composite readiness score that drives pacing decisions each epoch. Batches are
    always exposed in natural curriculum order (easiest first), starting from batch 0, and the
    active set grows monotonically.

    ## Design guarantees:
    - Data is introduced monotonically (active batch count never shrinks).
    - At least one batch increment is added per epoch until all data is active.
    - High readiness allows up to 3x the minimum floor in a single epoch.
    """

    def __init__(self,
        total_samples:      int,
        total_epochs:       int,
        batch_size:         int,
        start_fraction:     float = 0.3,
        window:             int =   5,
    ):
        """# Instantiate Adaptive Curriculum Pacing Schedule.

        ## Args:
            * total_samples     (int):      Total number of training samples.
            * total_epochs      (int):      Total number of training epochs.
            * batch_size        (int):      Number of samples per batch.
            * start_fraction    (float):    Fraction of batches exposed at epoch 1. Defaults to 0.3.
            * window            (int):      Lookback window for all four signals. Defaults to 5.
        """
        # Initialize protocol.
        super(AdaptiveSchedule, self).__init__(
            schedule_id =   "adaptive",
            total_samples = total_samples,
            total_epochs =  total_epochs,
            batch_size =    batch_size,
        )

        # Validate start fraction.
        if not 0.0 < start_fraction < 1.0:
            raise ValueError(
                f"start_fraction must be in (0, 1); got {start_fraction}"
            )

        # Define pacing state.
        self._start_fraction_:      float =     start_fraction
        self._window_:              int =       window
        self._active_batches_:      int =       max(1, int(self._total_batches_ * start_fraction))
        self._next_batch_to_add_:   int =       self._active_batches_

        # Define metric histories.
        self._loss_history_:        List[float] =   []
        self._val_acc_history_:     List[float] =   []
        self._batch_std_history_:   Dict =          {}
        self._batch_grad_history_:  Dict =          {}

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Adaptive Schedule Dictionary Representation"""
        return  {
                    **super().dict,
                    "start_fraction":   self._start_fraction_,
                    "window":           self._window_,
                }

    # HELPERS ======================================================================================

    @override
    def _order_(self,
        epoch:          int,
        loss:           float =     None,
        val_acc:        float =     None,
        std_df =        None,
        grad_norm_df =  None,
        **kwargs:       Any
    ) -> List[int]:
        """# Compute Adaptive Batch Ordering for Current Epoch.

        ## Args:
            * epoch         (int):              Current epoch (1-indexed).
            * loss          (float | None):     Training loss for this epoch.
            * val_acc       (float | None):     Validation accuracy for this epoch.
            * std_df        (DataFrame | None): Per-batch activation std DataFrame.
            * grad_norm_df  (DataFrame | None): Per-batch gradient norm DataFrame.
            * **kwargs      (Any):              Ignored.

        ## Returns:
            * List[int]:    Batch indices [0..active_batches] in natural curriculum order.
        """
        # Update metric histories.
        if loss is not None:    self._loss_history_.append(loss)
        if val_acc is not None: self._val_acc_history_.append(val_acc)

        # Update per-batch activation std history.
        if std_df is not None and not std_df.empty and "mean_std" in std_df.columns:
            for batch_idx in std_df.index:
                if batch_idx not in self._batch_std_history_:
                    self._batch_std_history_[batch_idx] = []
                self._batch_std_history_[batch_idx].append(
                    float(std_df.loc[batch_idx, "mean_std"])
                )

        # Update per-batch gradient norm history.
        if grad_norm_df is not None and not grad_norm_df.empty \
                and "mean_grad_norm" in grad_norm_df.columns:
            for batch_idx in grad_norm_df.index:
                if batch_idx not in self._batch_grad_history_:
                    self._batch_grad_history_[batch_idx] = []
                self._batch_grad_history_[batch_idx].append(
                    float(grad_norm_df.loc[batch_idx, "mean_grad_norm"])
                )

        # If all batches are already active, return full natural order.
        if self._next_batch_to_add_ >= self._total_batches_:
            return list(range(self._total_batches_))

        # Compute composite readiness score from all four signals.
        readiness:          float = self._readiness_()

        # Dynamic floor: guarantees all data is active by the final epoch.
        remaining_batches:  int =   self._total_batches_ - self._next_batch_to_add_
        remaining_epochs:   int =   self._total_epochs_ - epoch

        if remaining_epochs > 0:
            min_to_add: int =   max(1, ceil(remaining_batches / remaining_epochs))
        else:
            min_to_add: int =   remaining_batches

        # Max pace: readiness=1 allows up to 3x the minimum floor.
        max_to_add:     int =   min(remaining_batches, 3 * min_to_add)
        batches_to_add: int =   int(min_to_add + readiness * (max_to_add - min_to_add))
        batches_to_add: int =   max(1, min(batches_to_add, remaining_batches))

        # Advance the active batch count.
        self._active_batches_       += batches_to_add
        self._next_batch_to_add_    += batches_to_add

        # Debug pacing decision.
        self.__logger__.debug(
            f"Epoch {epoch}: readiness = {readiness:.4f}; "
            f"added {batches_to_add} batches; "
            f"active = {self._active_batches_}/{self._total_batches_}"
        )

        return list(range(self._active_batches_))

    def _readiness_(self) -> float:
        """# Compute Composite Readiness Score."""
        r_loss:     float = self._loss_plateau_signal_()
        r_val:      float = self._val_acc_trend_signal_()
        r_act:      float = self._activation_stability_signal_()
        r_grad:     float = self._gradient_norm_signal_()
        return (r_loss + r_val + r_act + r_grad) / 4.0

    def _loss_plateau_signal_(self) -> float:
        """# Loss Plateau Signal."""
        import numpy as np
        if len(self._loss_history_) < 2 * self._window_: return 0.0
        prior =             self._loss_history_[-(2 * self._window_):-self._window_]
        recent =            self._loss_history_[-self._window_:]
        mean_prior =        np.mean(prior)
        mean_recent =       np.mean(recent)
        if mean_prior < 1e-8: return 1.0
        relative_change =   (mean_recent - mean_prior) / mean_prior
        return float(1.0 / (1.0 + np.exp(-10.0 * relative_change)))

    def _val_acc_trend_signal_(self) -> float:
        """# Validation Accuracy Trend Signal."""
        import numpy as np
        if len(self._val_acc_history_) < self._window_: return 0.0
        recent =            self._val_acc_history_[-self._window_:]
        x =                 np.arange(len(recent), dtype=float)
        slope =             np.polyfit(x, recent, 1)[0]
        std_val =           np.std(recent)
        if std_val < 1e-8: return 1.0
        normalized_slope =  slope / std_val
        return float(1.0 / (1.0 + np.exp(5.0 * normalized_slope)))

    def _activation_stability_signal_(self) -> float:
        """# Activation Stability Signal."""
        if not self._batch_std_history_: return 0.0
        total, count = 0.0, 0
        for history in self._batch_std_history_.values():
            _, stability = self._detect_plateau_(history)
            total += stability; count += 1
        return total / count if count > 0 else 0.0

    def _gradient_norm_signal_(self) -> float:
        """# Gradient Norm Stability Signal."""
        if not self._batch_grad_history_: return 0.0
        total, count = 0.0, 0
        for history in self._batch_grad_history_.values():
            _, stability = self._detect_plateau_(history)
            total += stability; count += 1
        return total / count if count > 0 else 0.0

    def _detect_plateau_(self, values: List[float]):
        """# Detect Whether a Scalar Time Series Has Plateaued."""
        import numpy as np
        if len(values) < self._window_ + 2: return False, 0.0
        recent =            values[-self._window_:]
        full =              values[:]
        cv_recent =         np.std(recent) / (abs(np.mean(recent)) + 1e-8)
        cv_full =           np.std(full)   / (abs(np.mean(full))   + 1e-8)
        if cv_full < 1e-8: return True, 1.0
        ratio =             cv_recent / cv_full
        stability_score =   max(0.0, 1.0 - ratio)
        is_plateau =        ratio < 0.25
        return is_plateau, stability_score