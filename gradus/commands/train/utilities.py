"""# gradus.commands.train.utilities

Training process utilities & helpers functions.
"""
__all__ =   [
                "compute_accuracy",
            ]

from sklearn.metrics    import accuracy_score
from torch              import argmax, Tensor

def compute_accuracy(
    predictions:    Tensor,
    targets:        Tensor,
    precision:      int =       4
) -> float:
    """# Calculate Classification Accuracy.

    ## Args:
        * predictions   (Tensor):   Classification predictions.
        * targets       (Tensor):   Classification ground truths.
        * precision     (int):      Number of decimal places to round to. Defaults to 4.

    ## Returns:
        * float:    Number of correct predictions divided by total samples.
    """
    return  round(number = accuracy_score(
                y_true =    targets.cpu(),
                y_pred =    argmax(input = predictions, dim = 1).cpu().numpy(),
                normalize = False
            ) / targets.size(0), ndigits = precision)