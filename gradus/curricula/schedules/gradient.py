"""# gradus.curricula.schedules.gradient

Gradient-peak curriculum pacing schedule implementation.
"""

__all__ = ["GradientSchedule"]

from typing                                 import Any, Dict, List, override

from gradus.curricula.schedules.protocol    import Schedule
from gradus.registration                    import register_schedule

@register_schedule(
    id =    "gradient",
    tags =  ["pacing", "adaptive", "gradient-informed"]
)
class GradientSchedule(Schedule):
    """# Gradient-Peak Curriculum Pacing Schedule

    Reorders all curriculum batches each epoch so that the batches with the highest
    mean gradient norm - where the model is currently most confounded - are presented
    first. All batches are seen every epoch; only their order changes.

    Information-theoretic motivation:
        Maximum learning occurs when the model is maximally confounded. By presenting
        the most confounding batches first each epoch, the model receives its strongest
        gradient signal when its learning capacity is freshest, and consolidates with
        easier samples afterward.

    Design guarantees:
        - All batches are seen every epoch - DSI is always 0.0.
        - During cold start, batches are presented in natural curriculum order.
        - After cold start, batches are sorted by descending mean gradient norm.
        - Smoothing is applied across adjacent batches before sorting.
        - Batches with no gradient history retain their natural curriculum position.
    """

    def __init__(self,
        total_samples:      int,
        total_epochs:       int,
        batch_size:         int,
        start_fraction:     float = 0.3,
        smooth_window:      int =   5,
        cold_start_epochs:  int =   1,
    ):
        """# Instantiate Gradient-Peak Curriculum Pacing Schedule.

        ## Args:
            * total_samples     (int):      Total number of training samples.
            * total_epochs      (int):      Total number of training epochs.
            * batch_size        (int):      Number of samples per batch.
            * start_fraction    (float):    Unused - kept for CLI consistency with other schedules.
                                            Defaults to 0.3.
            * smooth_window     (int):      Number of adjacent batches to average gradient norms
                                            over before sorting. Defaults to 5.
            * cold_start_epochs (int):      Epochs to present batches in natural curriculum order
                                            before activating gradient-based reordering. Defaults 
                                            to 1.
        """
        # Initialize protocol.
        super(GradientSchedule, self).__init__(
            schedule_id =   "gradient",
            total_samples = total_samples,
            total_epochs =  total_epochs,
            batch_size =    batch_size,
        )

        # Validate parameters.
        if not 0.0 < start_fraction < 1.0:
            raise ValueError(f"start_fraction must be in (0, 1); got {start_fraction}")

        # Define properties.
        self._start_fraction_:      float =                     start_fraction
        self._smooth_window_:       int =                       smooth_window
        self._cold_start_epochs_:   int =                       cold_start_epochs

        # Gradient norm history: batch_idx -> list of per-epoch mean_grad_norm values.
        self._batch_grad_history_:  Dict[int, List[float]] =    {}

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Gradient Schedule Dictionary Representation"""
        return  {
                    **super().dict,
                    "smooth_window":        self._smooth_window_,
                    "cold_start_epochs":    self._cold_start_epochs_,
                }

    # HELPERS ======================================================================================

    @override
    def _order_(self,
        epoch:          int,
        grad_norm_df =  None,
        **kwargs:       Any
    ) -> List[int]:
        """# Compute Gradient-Peak Batch Ordering for Current Epoch.

        Returns all batch indices sorted by descending mean gradient norm - most
        confounding batches first. Falls back to natural curriculum order during
        cold start or when gradient history is unavailable.

        ## Args:
            * epoch         (int):              Current epoch (1-indexed).
            * grad_norm_df  (DataFrame | None): Per-batch gradient norm DataFrame, indexed by
                                                batch index with column 'mean_grad_norm'.
            * **kwargs      (Any):              Ignored.

        ## Returns:
            * List[int]:    All batch indices, sorted by descending mean gradient norm.
        """
        # Update gradient norm history from previous epoch.
        if  grad_norm_df is not None and not grad_norm_df.empty \
            and "mean_grad_norm" in grad_norm_df.columns:

            for batch_idx in grad_norm_df.index:

                if batch_idx not in self._batch_grad_history_:
                    self._batch_grad_history_[batch_idx] = []

                self._batch_grad_history_[batch_idx].append(
                    float(grad_norm_df.loc[batch_idx, "mean_grad_norm"])
                )

        # During cold start or insufficient history, use natural curriculum order.
        if epoch <= self._cold_start_epochs_ or not self._batch_grad_history_:
            return list(range(self._total_batches_))

        # Compute mean gradient norm per batch across all recorded epochs.
        mean_norms: List[float] =   [
                                        sum(self._batch_grad_history_.get(b, [0.0])) /
                                        max(1, len(self._batch_grad_history_.get(b, [0.0])))
                                        for b in range(self._total_batches_)
                                    ]

        # Apply smoothing across adjacent batches to reduce noise.
        smoothed:   List[float] =   self._smooth_(mean_norms)

        # Sort batch indices by descending smoothed gradient norm.
        # Batches with no history (norm=0.0) naturally fall to the end.
        order:      List[int] =     sorted(
                                        range(self._total_batches_),
                                        key =       lambda i: smoothed[i],
                                        reverse =   True
                                    )

        # Debug top and bottom of ordering.
        self.__logger__.debug(
            f"Epoch {epoch}: top-5 batches = {order[:5]}, "
            f"bottom-5 batches = {order[-5:]}"
        )

        return order

    def _smooth_(self,
        values: List[float]
    ) -> List[float]:
        """# Apply Rolling Mean Smoothing to Gradient Norm Values.

        ## Args:
            * values    (List[float]):  Raw per-batch gradient norm values.

        ## Returns:
            * List[float]:  Smoothed values of same length.
        """
        n:          int =           len(values)
        half:       int =           self._smooth_window_ // 2
        smoothed:   List[float] =   []

        for i in range(n):
            lo:     int =   max(0, i - half)
            hi:     int =   min(n, i + half + 1)
            smoothed.append(sum(values[lo:hi]) / (hi - lo))

        return smoothed