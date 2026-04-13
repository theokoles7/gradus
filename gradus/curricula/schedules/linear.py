"""# gradus.curricula.schedules.linear

Linear curriculum pacing schedule implementation.
"""

__all__ = ["LinearSchedule"]

from typing                                 import Any, Dict, override

from gradus.curricula.schedules.protocol    import Schedule
from gradus.registration                    import register_schedule

@register_schedule(
    id =    "linear",
    tags =  ["pacing", "fixed"]
)
class LinearSchedule(Schedule):
    """# Linear Curriculum Pacing Schedule

    Linearly ramps the fraction of training data exposed from
    `start_fraction` at epoch 1 to 1.0 by `full_data_epoch`, then
    holds at 1.0 for all remaining epochs.
    """

    def __init__(self,
        total_samples:      int,
        total_epochs:       int,
        start_fraction:     float = 0.3,
        full_data_epoch:    int =   None
    ):
        """# Instantiate Linear Curriculum Pacing Schedule.

        ## Args:
            * total_samples     (int):      Total number of training samples.
            * total_epochs      (int):      Total number of training epochs.
            * start_fraction    (float):    Fraction of data exposed at epoch 1. Defaults to 0.3.
            * full_data_epoch   (int):      Epoch at which 100% of data is first exposed. Defaults 
                                            to 60% of total_epochs.
        """
        # Initialize protocol.
        super(LinearSchedule, self).__init__(
            schedule_id =   "linear",
            total_samples = total_samples,
            total_epochs =  total_epochs
        )

        # Validate start fraction.
        if not 0.0 < start_fraction < 1.0:
            raise ValueError(
                f"start_fraction must be in (0, 1); got {start_fraction}"
            )

        # Define properties.
        self._start_fraction_:  float = start_fraction
        self._full_data_epoch_: int =   full_data_epoch if full_data_epoch is not None \
                                        else max(1, int(0.6 * total_epochs))

        # Validate full_data_epoch.
        if not 1 <= self._full_data_epoch_ <= total_epochs:
            raise ValueError(
                f"full_data_epoch must be in [1, {total_epochs}]; "
                f"got {self._full_data_epoch_}"
            )

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Linear Schedule Dictionary Representation"""
        return  {
                    **super().dict,
                    "start_fraction":   self._start_fraction_,
                    "full_data_epoch":  self._full_data_epoch_
                }

    # HELPERS ======================================================================================

    @override
    def _fraction_(self,
        epoch:      int,
        **metrics:  Any
    ) -> float:
        """# Compute Linear Data Fraction for Current Epoch.

        ## Args:
            * epoch     (int):  Current epoch (1-indexed).
            * **metrics (Any):  Ignored — linear schedule is purely epoch-driven.

        ## Returns:
            * float:    Fraction of training data to expose.
        """
        # Once full_data_epoch is reached, expose all data.
        if epoch >= self._full_data_epoch_: return 1.0

        # Otherwise, linearly interpolate between start_fraction and 1.0.
        return  self._start_fraction_ + (1.0 - self._start_fraction_) * (
                    epoch / self._full_data_epoch_
                )