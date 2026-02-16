"""# gradus.metrics.saturation_time

Measurement of the time required for a model's weights to saturate/converge for an individual image 
sample.
"""

__all__ =   [
                "TimeToSaturation",
                "time_to_saturation",
            ]

from typing         import Dict, List

from torch          import Tensor
from torch.nn       import CrossEntropyLoss, Module
from torch.optim    import SGD

class TimeToSaturation():
    """# Time-to-Saturation Measurement"""

    def __init__(self,
        # Sample & Model
        sample:         Tensor,
        target:         Tensor,
        model:          Module, *,

        # Calculation parameters
        max_iterations: int =   200,
        threshold:      float = 1e-5,
        window:         int =   10,
        learning_rate:  float = 0.01
    ):
        """# Calculate Sample's Time-to-Saturation Metric.

        ## Args:
            * sample            (Tensor):   Sample whose saturation time is being measured.
            * target            (Tensor):   Sample's corresponding target tensor/label.
            * model             (Module):   Model by whom saturation time will be determined.
            * max_iterations    (int):      Maximum number of iterations allowed before abandoning 
                                            measurement attempt. Defaults to 200.
            * threshold         (float):    Threshold under which the delta of the weights' L2 norms 
                                            must fall to be considered "saturated". Defaults to 
                                            1e-5.
            * window            (int):      Number of consecutive iterations for which delta must 
                                            remain under threshold to achieve "stable saturation". 
                                            Defaults to 10.
            * learning_rate     (float):    Learning rate with which optimizer will be configured. 
                                            Defaults to 0.01.
        """
        # Define properties.
        self._sample_:                  Tensor =                    sample
        self._target_:                  Tensor =                    target
        self._model_:                   Module =                    model
        self._max_iterations_:          int =                       max_iterations
        self._threshold_:               float =                     threshold
        self._window_:                  int =                       window
        self._learning_rate_:           float =                     learning_rate

        # Discover learnable layers of network.
        self._learnable_layers_:        Dict[str, Module] =         self._get_learnable_layers_(
                                                                        model = model
                                                                    )

        # Initialize metric tracking.
        self._iteration_:               int =                       0
        self._saturated_:               bool =                      False
        self._loss_history_:            List[float] =               []
        self._weight_delta_history_:    Dict[str, List[float]] =    {
                                                                        name: []
                                                                        for name 
                                                                        in self._learnable_layers_
                                                                    }
        self._stable_counts_:           Dict[str, int] =            {
                                                                        name: 0
                                                                        for name
                                                                        in self._learnable_layers_
                                                                    }
        self._layer_saturated_:         Dict[str, bool] =           {
                                                                        name: False
                                                                        for name
                                                                        in self._learnable_layers_
                                                                    }
        self._layer_saturation_iters_:  Dict[str, int] =            {
                                                                        name: -1
                                                                        for name
                                                                        in self._learnable_layers_
                                                                    }
        
        # Calculate metric.
        self._calculate_()

    # PROPERTIES ===================================================================================

    @property
    def final_loss(self) -> float:
        """# Loss Value at Final Iteration"""
        return self._loss_history_[-1] if self._loss_history_ else float("nan")

    @property
    def iterations(self) -> int:
        """# Number of Iterations Executed"""
        return self._iteration_
    
    @property
    def layer_saturation_iters(self) -> Dict[str, int]:
        """Iteration at which Each Layer First Saturated"""
        return self._layer_saturation_iters_
    
    @property
    def learnable_layer_names(self) -> List[str]:
        """# Names of Discovered Learnable Layers"""
        return list(self._learnable_layers_.keys())
    
    @property
    def loss_history(self) -> List[float]:
        """# Loss Value at Each Iteration"""
        return self._loss_history_
    
    @property
    def saturated(self) -> bool:
        """# Are All Layers Saturated?"""
        return self._saturated_
    
    @property
    def weight_delta_history(self) -> Dict[str, List[float]]:
        """# Layer-Wise Weight Delta at Each Iteration"""
        return self._weight_delta_history_

    # HELPERS ======================================================================================

    def _calculate_(self) -> None:
        """# Calculate Time-to-Saturation of Sample."""
        # Define loss criteria & optimizer.
        loss_fn:        CrossEntropyLoss =  CrossEntropyLoss()
        optimizer:      SGD =               SGD(
                                                params =    self._model_.parameters(),
                                                lr =        self._learning_rate_
                                            )

        # Prepare single-sample batch.
        sample_batch:   Tensor =            self._sample_.unsqueeze(0)
        target_batch:   Tensor =            self._target_.unsqueeze(0)      \
                                                if self._target_.dim() == 0 \
                                                else self._target_.view(1)
        
        # Take note of initial model weights.
        prev_weights:   Dict[str, Tensor] = self._snapshot_weights_()

        # Place model in training mode.
        self._model_.train()

        # For no more than the maximum allowed iterations...
        for self._iteration_ in range(1, self._max_iterations_ + 1):

            # Forward pass.
            prediction:     Tensor =            self._model_(sample_batch)
            loss:           Tensor =            loss_fn(prediction, target_batch)

            # Back propagation.
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Record loss.
            self._loss_history_.append(loss.item())

            # Compute weight deltas.
            deltas:         Dict[str, float] =  self._compute_weight_delta_(
                                                    prev_weights = prev_weights
                                                )
            
            # Compute layer-wise stabilization.
            self._compute_layer_stability_(deltas = deltas, iteration = self._iteration_)

            # Snapshot weights for next iteration.
            prev_weights:   Dict[str, Tensor] = self._snapshot_weights_()

            # However, if all layers are now saturated, we are done.
            if all(self._layer_saturated_.values()): self._saturated_ = True; break

    def _compute_layer_stability_(self,
        deltas:     Dict[str, float],
        iteration:  int
    ) -> None:
        """# Compute Layer-Wise Stabilization.

        ## Args:
            * deltas    (Dict[str, float]): Layer-wise weight deltas for current iteration.
            * iteration (int):              Current iteration.
        """
        # For each learnable layer...
        for layer in self._learnable_layers_:

            # Record delta history.
            self._weight_delta_history_[layer].append(deltas[layer])

            # If layer is not already confirmed to be saturated...
            if not self._layer_saturated_[layer]:

                # If new delta passes (is less than) threshold...
                if deltas[layer] < self._threshold_:

                    # Increment stable iterations count for layer.
                    self._stable_counts_[layer] += 1

                    # If layer has reached complete saturation (according to window)...
                    if self._stable_counts_[layer] >= self._window_:

                        # Mark layer as completely saturated & record iteration at which 
                        # stabilization began.
                        self._layer_saturated_[layer] =         True
                        self._layer_saturation_iters_[layer] =  iteration - self._window_ + 1
                
                # Otherwise, reset stable iteration count for this layer.
                else:   self._stable_counts_[layer] = 0

    def _compute_weight_delta_(self,
        prev_weights:       Dict[str, Tensor]
    ) -> Dict[str, float]:
        """# Compute L2 Norm of Weight Delta(s).

        ## Args:
            * prev_weights  (Dict[str, Tensor]):    Previous weight snapshot being compared against.

        ## Returns:
            * Dict[str, float]: Mapping of layer names to the L2 norm of their weight deltas.
        """
        return  {
                    name: (layer.weight.data - prev_weights[name]).norm(p = "fro").item()
                    for name, layer in self._learnable_layers_.items()
                }

    def _get_learnable_layers_(self,
        model:  Module
    ) -> Dict[str, Module]:
        """# Get Learnable Layers of Model.

        ## Args:
            * model (Module):   Network to inspect.

        ## Returns:
            * Dict[str, Module]:    Mapping of layer names to their modules.
        """
        return  {
                    name: layer
                    for name, layer in model.named_modules()
                    if name
                        and hasattr(layer, "weight")
                        and layer.weight is not None
                        and layer.weight.requires_grad
                }

    def _snapshot_weights_(self) -> Dict[str, Tensor]:
        """# Capture Clone of each Learnable Layer.

        ## Returns:
            * Dict[str, Tensor]:    Cloned weight tensors keyed by layer name.
        """
        return  {
                    name: layer.weight.data.clone()
                    for name, layer in self._learnable_layers_.items()
                }
    

# QUICK-ACCESS UTILITY =============================================================================

def time_to_saturation(
    # Sample & Model
    sample:         Tensor,
    target:         Tensor,
    model:          Module, *,

    # Calculation Parameters
    max_iterations: int =   200,
    threshold:      float = 1e-5,
    window:         int =   10,
    learning_rate:  float = 0.01
) -> int:
    """# Calculate Sample's Time-to-Saturation Metric.

    ## Args:
        * sample            (Tensor):   Sample whose saturation time is being measured.
        * target            (Tensor):   Sample's corresponding target tensor/label.
        * model             (Module):   Model by whom saturation time will be determined.
        * max_iterations    (int):      Maximum number of iterations allowed before abandoning 
                                        measurement attempt. Defaults to 200.
        * threshold         (float):    Threshold under which the delta of the weights' L2 norms 
                                        must fall to be considered "saturated". Defaults to 
                                        1e-5.
        * window            (int):      Number of consecutive iterations for which delta must 
                                        remain under threshold to achieve "stable saturation". 
                                        Defaults to 10.
        * learning_rate     (float):    Learning rate with which optimizer will be configured. 
                                        Defaults to 0.01.

    ## Returns:
        * int:  Number of iterations required for sample saturation.
    """
    return TimeToSaturation(**locals()).iterations