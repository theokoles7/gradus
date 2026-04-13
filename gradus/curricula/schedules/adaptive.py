"""# gradus.curricula.schedules.adaptive

Adaptive curriculum pacing schedule implementation.
"""

__all__ = ["AdaptiveSchedule"]

from typing                                 import Any, Dict, List, override

from gradus.curricula.schedules.protocol    import Schedule
from gradus.registration                    import register_schedule

@register_schedule(
    id =    "adaptive",
    tags =  ["pacing", "adaptive"]
)
class AdaptiveSchedule(Schedule):
    """# Adaptive Curriculum Pacing Schedule

    Controls the rate at which new curriculum data is introduced by monitoring
    four training signals — loss plateau, validation accuracy trend, activation
    stability, and gradient norm stability — and computing a composite readiness
    score that drives pacing decisions each epoch.

    Design guarantees:
        - Data is introduced monotonically (active fraction never shrinks).
        - At least one sample increment is added per epoch until all data is active.
        - High readiness allows up to 3x the minimum floor in a single epoch.

    Signals default to 0.0 (conservative/cold-start) when insufficient history
    is available or when activation/gradient data is not supplied by the training
    loop. This means the schedule degrades gracefully to minimum-pace advancement
    until all four signals have enough history to make informed decisions.

    Constants:
        window = 5      : lookback window for all signals
        0.25            : plateau threshold in _detect_plateau_()
        sigmoid k = 10  : loss signal transition sharpness (scale-independent)
        sigmoid k = 5   : val-acc signal transition sharpness (scale-independent)
        3x multiplier   : max pace = 3x minimum floor (structural choice)
    """

    def __init__(self,
        total_samples:      int,
        total_epochs:       int,
        start_fraction:     float = 0.3,
        window:             int =   5
    ):
        """# Instantiate Adaptive Curriculum Pacing Schedule.

        ## Args:
            * total_samples     (int):      Total number of training samples.
            * total_epochs      (int):      Total number of training epochs.
            * start_fraction    (float):    Fraction of data exposed at epoch 1. Defaults to 0.3.
            * window            (int):      Lookback window for all four signals. Defaults to 5.
        """
        # Initialize protocol.
        super(AdaptiveSchedule, self).__init__(
            schedule_id =   "adaptive",
            total_samples = total_samples,
            total_epochs =  total_epochs
        )

        # Validate start fraction.
        if not 0.0 < start_fraction < 1.0:
            raise ValueError(
                f"start_fraction must be in (0, 1); got {start_fraction}"
            )

        # Define pacing state.
        self._start_fraction_:      float =         start_fraction
        self._window_:              int =           window
        self._active_samples_:      int =           max(1, int(total_samples * start_fraction))
        self._next_sample_to_add_:  int =           self._active_samples_

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
                    "window":           self._window_
                }

    # HELPERS ======================================================================================

    @override
    def _fraction_(self,
        epoch:          int,
        loss:           float =             None,
        val_acc:        float =             None,
        std_df =        None,
        grad_norm_df =  None,
        **kwargs:       Any
    ) -> float:
        """# Compute Adaptive Data Fraction for Current Epoch.

        ## Args:
            * epoch         (int):              Current epoch (1-indexed).
            * loss          (float | None):     Training loss for this epoch.
            * val_acc       (float | None):     Validation accuracy for this epoch.
            * std_df        (DataFrame | None): Per-batch activation std DataFrame, indexed by
                                                curriculum batch index with column 'mean_std'.
                                                Defaults to None (activation signal = 0.0).
            * grad_norm_df  (DataFrame | None): Per-batch gradient norm DataFrame, indexed by
                                                curriculum batch index with column 'mean_grad_norm'.
                                                Defaults to None (gradient signal = 0.0).
            * **kwargs      (Any):              Ignored.

        ## Returns:
            * float:    Fraction of training data to expose.
        """
        from math   import ceil
        from pandas import DataFrame

        # Update metric histories.
        if loss is not None:        self._loss_history_.append(loss)
        if val_acc is not None:     self._val_acc_history_.append(val_acc)

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

        # If all data is already active, return 1.0 immediately.
        if self._next_sample_to_add_ >= self._total_samples_:
            return 1.0

        # Compute composite readiness score from all four signals.
        readiness:          float =     self._readiness_()

        # Dynamic floor: guarantees all data is active by the final epoch.
        remaining_samples:  int =       self._total_samples_ - self._next_sample_to_add_
        remaining_epochs:   int =       self._total_epochs_ - epoch

        if remaining_epochs > 0:
            min_to_add: int =   max(1, ceil(remaining_samples / remaining_epochs))
        else:
            min_to_add: int =   remaining_samples

        # Max pace: readiness=1 allows up to 3x the minimum floor.
        max_to_add:     int =   min(remaining_samples, 3 * min_to_add)
        samples_to_add: int =   int(min_to_add + readiness * (max_to_add - min_to_add))
        samples_to_add: int =   max(1, min(samples_to_add, remaining_samples))

        # Advance the active window.
        self._active_samples_       += samples_to_add
        self._next_sample_to_add_   += samples_to_add

        # Debug pacing decision.
        self.__logger__.debug(
            f"Epoch {epoch}: readiness = {readiness:.4f}; "
            f"added {samples_to_add} samples; "
            f"active = {self._active_samples_}/{self._total_samples_}"
        )

        return self._active_samples_ / self._total_samples_

    def _readiness_(self) -> float:
        """# Compute Composite Readiness Score.

        Averages four signals — loss plateau, validation accuracy trend,
        activation stability, and gradient norm stability — each in [0, 1].
        Returns 0.0 when insufficient history is available.

        ## Returns:
            * float:    Composite readiness score in [0, 1].
        """
        r_loss:     float = self._loss_plateau_signal_()
        r_val:      float = self._val_acc_trend_signal_()
        r_act:      float = self._activation_stability_signal_()
        r_grad:     float = self._gradient_norm_signal_()

        return (r_loss + r_val + r_act + r_grad) / 4.0

    def _loss_plateau_signal_(self) -> float:
        """# Loss Plateau Signal.

        High score means training loss has stopped decreasing. Compares mean
        loss over the most recent `window` epochs to the preceding `window`
        epochs via relative change. Sigmoid steepness = 10.

        ## Returns:
            * float:    Signal in [0, 1].
        """
        import numpy as np

        # Insufficient history — conservative default.
        if len(self._loss_history_) < 2 * self._window_: return 0.0

        prior:          List[float] =   self._loss_history_[-(2 * self._window_):-self._window_]
        recent:         List[float] =   self._loss_history_[-self._window_:]

        mean_prior:     float =         np.mean(prior)
        mean_recent:    float =         np.mean(recent)

        # Loss already near zero — fully plateaued.
        if mean_prior < 1e-8: return 1.0

        # Positive relative change → loss got worse or flat → high signal.
        relative_change:    float = (mean_recent - mean_prior) / mean_prior
        return float(1.0 / (1.0 + np.exp(-10.0 * relative_change)))

    def _val_acc_trend_signal_(self) -> float:
        """# Validation Accuracy Trend Signal.

        High score means val accuracy has plateaued or is declining. Fits a
        linear trend over the recent `window` epochs, normalized by within-window
        std for scale independence. Sigmoid steepness = 5.

        ## Returns:
            * float:    Signal in [0, 1].
        """
        import numpy as np

        # Insufficient history — conservative default.
        if len(self._val_acc_history_) < self._window_: return 0.0

        recent:             List[float] =   self._val_acc_history_[-self._window_:]
        x:                  np.ndarray =    np.arange(len(recent), dtype = float)
        slope:              float =         np.polyfit(x, recent, 1)[0]
        std_val:            float =         np.std(recent)

        # Perfectly flat — fully plateaued.
        if std_val < 1e-8: return 1.0

        # Positive slope (still improving) → low signal.
        normalized_slope:   float = slope / std_val
        return float(1.0 / (1.0 + np.exp(5.0 * normalized_slope)))

    def _activation_stability_signal_(self) -> float:
        """# Activation Stability Signal.

        High score means forward-pass activation stds for active curriculum
        batches have plateaued — the model represents them consistently.
        Returns 0.0 when no activation history is available.

        ## Returns:
            * float:    Signal in [0, 1].
        """
        # No activation data supplied by training loop yet.
        if not self._batch_std_history_: return 0.0

        total:  float = 0.0
        count:  int =   0

        for history in self._batch_std_history_.values():
            _, stability = self._detect_plateau_(history)
            total += stability
            count += 1

        return total / count if count > 0 else 0.0

    def _gradient_norm_signal_(self) -> float:
        """# Gradient Norm Stability Signal.

        High score means gradient L2 norms for active batches have plateaued —
        the model is no longer learning much from them. Structurally identical
        to activation stability but measures the backward pass independently.
        Returns 0.0 when no gradient norm history is available.

        ## Returns:
            * float:    Signal in [0, 1].
        """
        # No gradient norm data supplied by training loop yet.
        if not self._batch_grad_history_: return 0.0

        total:  float = 0.0
        count:  int =   0

        for history in self._batch_grad_history_.values():
            _, stability = self._detect_plateau_(history)
            total += stability
            count += 1

        return total / count if count > 0 else 0.0

    def _detect_plateau_(self,
        values: List[float]
    ):
        """# Detect Whether a Scalar Time Series Has Plateaued.

        Uses self-calibrating comparison: the coefficient of variation (CV)
        over the recent `window` steps is compared to the CV over the full
        history. If the recent CV is less than 25% of the historical CV,
        the series is considered to have plateaued. This avoids hardcoded
        delta thresholds — the decision is always relative to the signal's
        own historical fluctuation range.

        ## Args:
            * values    (List[float]):  Full time series.

        ## Returns:
            * Tuple[bool, float]:   (is_plateau, stability_score in [0, 1]).
        """
        import numpy as np

        # Insufficient history.
        if len(values) < self._window_ + 2: return False, 0.0

        recent:         List[float] =   values[-self._window_:]
        full:           List[float] =   values[:]

        cv_recent:  float = np.std(recent) / (abs(np.mean(recent)) + 1e-8)
        cv_full:    float = np.std(full)   / (abs(np.mean(full))   + 1e-8)

        # Full series is already flat.
        if cv_full < 1e-8: return True, 1.0

        ratio:              float = cv_recent / cv_full
        stability_score:    float = max(0.0, 1.0 - ratio)
        is_plateau:         bool =  ratio < 0.25

        return is_plateau, stability_score