"""# gradus.curricula.metrics.model_informed.convergence_time

Measurement of the time required for a model's loss to converge for an individual image sample.
"""

__all__ =   [
                "TimeToConvergence",
                "time_to_convergence",
            ]

from typing             import List, Union

from torch              import device as t_device, Tensor
from torch.nn           import MSELoss
from torch.optim        import SGD

from gradus.networks    import Autoencoder
from gradus.utilities   import determine_device

class TimeToConvergence():
    """# Time-to-Convergence Measurement"""

    def __init__(self,
        # Sample
        sample:         Tensor, *,

        # Calculation parameters
        max_iterations: int =                   1000,
        threshold:      float =                 1e-3,
        window:         int =                   5,
        learning_rate:  float =                 0.05,
        device:         Union[str, t_device] =  "auto"
    ):
        """# Calculate Sample's Time-to-Convergence Metric.

        ## Args:
            * sample            (Tensor):       Sample whose convergence time is being measured.
            * max_iterations    (int):          Maximum number of iterations allowed before 
                                                abandoning measurement attempt. Defaults to 1000.
            * threshold         (float):        Threshold under which the loss delta must fall to be 
                                                considered "converged". Defaults to 1e-3.
            * window            (int):          Number of consecutive iterations for which loss 
                                                delta must remain under threshold to achieve "stable 
                                                convergence". Defaults to 5.
            * learning_rate     (float):        Learning rate with which optimizer will be 
                                                configured. Defaults to 0.05.
            * device            (str | device): Torch computation device. Defaults to "auto".
        """
        # Define properties.
        self._device_:              t_device =      determine_device(device = device)
        self._sample_:              Tensor =        sample.to(self._device_)
        self._max_iterations_:      int =           max_iterations
        self._threshold_:           float =         threshold
        self._window_:              int =           window
        self._learning_rate_:       float =         learning_rate
        self._input_channels_:      int =           1 if sample.dim() == 2 else sample.shape[0]
        self._model_:               Autoencoder =   Autoencoder(
                                                        channels =  self._input_channels_
                                                    ).to(self._device_)

        # Initialize metric tracking.
        self._iteration_:           int =           0
        self._converged_:           bool =          False
        self._loss_history_:        List[float] =   []
        self._loss_delta_history_:  List[float] =   []

        # Calculate metric.
        self._calculate_()

    # PROPERTIES ===================================================================================

    @property
    def converged(self) -> bool:
        """# Did Loss Converge Within Max Iterations?"""
        return self._converged_

    @property
    def final_loss(self) -> float:
        """# Loss Value at Final Iteration"""
        return self._loss_history_[-1] if self._loss_history_ else float("nan")

    @property
    def iterations(self) -> int:
        """# Number of Iterations Executed"""
        return self._iteration_

    @property
    def loss_delta_history(self) -> List[float]:
        """# Loss Delta at Each Iteration"""
        return self._loss_delta_history_

    @property
    def loss_history(self) -> List[float]:
        """# Loss Value at Each Iteration"""
        return self._loss_history_

    # HELPERS ======================================================================================

    def _calculate_(self) -> None:
        """# Calculate Time-to-Convergence of Sample."""
        # Define loss criterion & optimizer.
        loss_fn:        MSELoss =   MSELoss()
        optimizer:      SGD =       SGD(
                                        params =    self._model_.parameters(),
                                        lr =        self._learning_rate_
                                    )

        # Prepare single-sample batch.
        sample_batch:   Tensor =    self._sample_.unsqueeze(0)

        # Place model in training mode.
        self._model_.train()

        # Track previous loss & consecutive stable iterations.
        prev_loss:      float =     None
        stable_count:   int =       0

        # For no more than the maximum allowed iterations...
        for self._iteration_ in range(1, self._max_iterations_ + 1):

            # Forward pass.
            reconstruction: Tensor =    self._model_(sample_batch)
            loss:           Tensor =    loss_fn(reconstruction, sample_batch)

            # Back propagation.
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Get loss calculation.
            current_loss:   float =     loss.item()

            # Record loss.
            self._loss_history_.append(current_loss)

            # After the first iteration...
            if prev_loss is not None:

                # Compute the delta between current and previous loss.
                delta:  float = abs(current_loss - prev_loss)

                # Record delta for history tracking.
                self._loss_delta_history_.append(delta)

                # If delta is less than threshold...
                if delta < self._threshold_:

                    # Increment stable iteration count.
                    stable_count += 1

                    # If stable count has reached window, convergence is achieved.
                    if stable_count >= self._window_: self._converged_ = True; break

                # Otherwise, reset stable iteration count.
                else: stable_count = 0

            # Update previous loss tracker.
            prev_loss = current_loss


# QUICK-ACCESS UTILITY =============================================================================

from gradus.curricula.metrics.model_informed.convergence_time.__args__  import ConvergenceTimeConfig
from gradus.registration                                                import register_metric

@register_metric(
    id =        "convergence-time",
    cls =       TimeToConvergence,
    config =    ConvergenceTimeConfig,
    tags =      ["model-informed"]
)
def time_to_convergence(
    # Sample
    sample:         Tensor, *,

    # Calculation parameters
    max_iterations: int =                   1000,
    threshold:      float =                 1e-3,
    window:         int =                   5,
    learning_rate:  float =                 0.05,
    device:         Union[str, t_device] =  "auto"
) -> int:
    """# Calculate Sample's Time-to-Convergence Metric.

    ## Args:
        * sample            (Tensor):       Sample whose convergence time is being measured.
        * max_iterations    (int):          Maximum number of iterations allowed before 
                                            abandoning measurement attempt. Defaults to 1000.
        * threshold         (float):        Threshold under which the loss delta must fall to be 
                                            considered "converged". Defaults to 1e-3.
        * window            (int):          Number of consecutive iterations for which loss 
                                            delta must remain under threshold to achieve "stable 
                                            convergence". Defaults to 5.
        * learning_rate     (float):        Learning rate with which optimizer will be 
                                            configured. Defaults to 0.05.
        * device            (str | device): Torch computation device. Defaults to "auto".

    ## Returns:
        * int:  Number of iterations required for loss convergence.
    """
    return TimeToConvergence(**locals()).iterations