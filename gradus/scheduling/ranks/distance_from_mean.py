"""# gradus.scheduling.ranks.distance_from_mean

Distance from mean ranking implementation.
"""

__all__ = ["distance_from_mean"]

from typing                 import List

from pandas                 import DataFrame

from gradus.registration    import register_rank

@register_rank(
    id =    "distance-from-mean",
    tags =  ["deviation", "distance-based"]
)
def distance_from_mean(
    metric: str,
    scores: DataFrame
) -> List[int]:
    """# Sort Indices by Absolute Deviation from Dataset Mean Score.

    ## Args:
        * metric    (str):          Metric by which sample indices should be ranked.
        * scores    (DataFrame):    Metric scores sheet.

    ## Returns:
        * List[int]:    Indices ranked by absolute distance from the mean.
    """
    # Compute mean score value.
    mean: float = scores[metric].mean()
 
    # Provide indices in order of absolute distance from the calculated mean.
    return  scores.assign(distance = (scores[metric] - mean).abs()) \
            .sort_values("distance")["index"].tolist()