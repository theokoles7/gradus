"""# gradus.curricula.schedules.uncertainty

Uncertainty-peak curriculum pacing schedule implementation.
"""

__all__ = ["UncertaintySchedule"]

from typing                                 import Any, Dict, List, override

from gradus.curricula.schedules.protocol    import Schedule
from gradus.registration                    import register_schedule

@register_schedule(
    id =    "uncertainty",
    tags =  ["pacing", "adaptive", "uncertainty-informed"]
)
class UncertaintySchedule(Schedule):
    """# Uncertainty-Peak Curriculum Pacing Schedule

    Reorders all curriculum batches each epoch so that the batches with the highest
    mean softmax entropy - where the model is currently most uncertain - are presented
    first. All batches are seen every epoch; only their order changes.

    Motivation:
        Softmax entropy directly measures classification uncertainty: a high-entropy
        batch is one the model has not yet resolved into confident predictions. By
        presenting the most uncertain batches first each epoch, the model focuses its
        freshest learning capacity on the samples it understands least. As uncertainty
        decreases, batches recede naturally toward the end of the epoch.

        This is the dynamic analogue of saturation-time: where saturation-time measures
        how long a dedicated autoencoder takes to stop updating on a sample (static,
        pre-training), uncertainty measures how long the classifier takes to stop being
        confused about a batch (dynamic, updated each epoch).

    Design guarantees:
        - All batches are seen every epoch - DSI is always 0.0.
        - During cold start, batches are presented in natural curriculum order.
        - After cold start, batches are sorted by descending mean softmax entropy.
        - Smoothing is applied across adjacent batches before sorting.
        - Batches with no entropy history retain their natural curriculum position.
    """

    def __init__(self,
        total_samples:      int,
        total_epochs:       int,
        batch_size:         int,
        start_fraction:     float = 0.3,
        smooth_window:      int =   5,
        cold_start_epochs:  int =   1,
    ):
        """# Instantiate Uncertainty-Peak Curriculum Pacing Schedule.

        ## Args:
            * total_samples     (int):      Total number of training samples.
            * total_epochs      (int):      Total number of training epochs.
            * batch_size        (int):      Number of samples per batch.
            * start_fraction    (float):    Unused - kept for CLI consistency with other schedules.
                                            Defaults to 0.3.
            * smooth_window     (int):      Number of adjacent batches to average entropy values
                                            over before sorting. Defaults to 5.
            * cold_start_epochs (int):      Epochs to present batches in natural curriculum order
                                            before activating uncertainty-based reordering.
                                            Defaults to 1.
        """
        # Initialize protocol.
        super(UncertaintySchedule, self).__init__(
            schedule_id =   "uncertainty",
            total_samples = total_samples,
            total_epochs =  total_epochs,
            batch_size =    batch_size,
        )

        # Validate parameters.
        if not 0.0 < start_fraction < 1.0:
            raise ValueError(f"start_fraction must be in (0, 1); got {start_fraction}")

        # Define properties.
        self._start_fraction_:          float =                     start_fraction
        self._smooth_window_:           int =                       smooth_window
        self._cold_start_epochs_:       int =                       cold_start_epochs

        # Entropy history: batch_idx -> list of per-epoch mean_entropy values.
        self._batch_entropy_history_:   Dict[int, List[float]] =    {}

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Uncertainty Schedule Dictionary Representation"""
        return  {
                    **super().dict,
                    "smooth_window":        self._smooth_window_,
                    "cold_start_epochs":    self._cold_start_epochs_,
                }

    # HELPERS ======================================================================================

    @override
    def _order_(self,
        epoch:          int,
        entropy_df =    None,
        **kwargs:       Any
    ) -> List[int]:
        """# Compute Uncertainty-Peak Batch Ordering for Current Epoch.

        Returns all batch indices sorted by descending mean softmax entropy - most
        uncertain batches first. Falls back to natural curriculum order during cold
        start or when entropy history is unavailable.

        ## Args:
            * epoch         (int):              Current epoch (1-indexed).
            * entropy_df    (DataFrame | None): Per-batch entropy DataFrame, indexed by
                                                batch index with column 'mean_entropy'.
            * **kwargs      (Any):              Ignored.

        ## Returns:
            * List[int]:    All batch indices, sorted by descending mean softmax entropy.
        """
        # Update entropy history from previous epoch.
        if  entropy_df is not None and not entropy_df.empty \
            and "mean_entropy" in entropy_df.columns:

            for batch_idx in entropy_df.index:

                if batch_idx not in self._batch_entropy_history_:
                    self._batch_entropy_history_[batch_idx] = []

                self._batch_entropy_history_[batch_idx].append(
                    float(entropy_df.loc[batch_idx, "mean_entropy"])
                )

        # During cold start or insufficient history, use natural curriculum order.
        if epoch <= self._cold_start_epochs_ or not self._batch_entropy_history_:
            return list(range(self._total_batches_))

        # Compute mean entropy per batch across all recorded epochs.
        mean_entropies: List[float] =   [
                                            sum(self._batch_entropy_history_.get(b, [0.0])) /
                                            max(1, len(self._batch_entropy_history_.get(b, [0.0])))
                                            for b in range(self._total_batches_)
                                        ]

        # Apply smoothing across adjacent batches to reduce noise.
        smoothed:       List[float] =   self._smooth_(mean_entropies)

        # Sort batch indices by descending smoothed entropy.
        # Batches with no history (entropy=0.0) naturally fall to the end.
        order:          List[int] =     sorted(
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
        """# Apply Rolling Mean Smoothing to Entropy Values.

        ## Args:
            * values    (List[float]):  Raw per-batch entropy values.

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