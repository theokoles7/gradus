"""# gradus.scheduling.ranks.descending

Descending ranking implementation.
"""

__all__ = ["descending"]

from typing                 import List

from pandas                 import DataFrame

from gradus.registration    import register_rank

@register_rank(
    id =    "descending",
    tags =  ["monotonic"]
)
def descending(
    metric: str,
    scores: DataFrame
) -> List[int]:
    """# Sort Indices in Descending Order According to Specified Metric.

    ## Args:
        * metric    (str):          Metric by which sample indices should be ranked.
        * scores    (DataFrame):    Metric scores sheet.

    ## Returns:
        * List[int]:    Indices ranked by metric in descending order.
    """
    return scores.sort_values(by = metric, ascending = False)["index"].tolist()