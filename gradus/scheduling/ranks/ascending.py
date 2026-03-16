"""# gradus.scheduling.ranks.ascending

Ascending ranking implementation.
"""

__all__ = ["ascending"]

from typing                 import List

from pandas                 import DataFrame

from gradus.registration    import register_rank

@register_rank(
    id =    "ascending",
    tags =  ["monotonic"]
)
def ascending(
    metric: str,
    scores: DataFrame
) -> List[int]:
    """# Sort Indices in Ascending Order According to Specified Metric.

    ## Args:
        * metric    (str):          Metric by which sample indices should be ranked.
        * scores    (DataFrame):    Metric scores sheet.

    ## Returns:
        * List[int]:    Indices ranked by metric in ascending order.
    """
    return scores.sort_values(by = metric)["index"].tolist()