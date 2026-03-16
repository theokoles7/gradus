"""# gradus.commands.analyze_scores.main

Main process entry point for analyze-scores command.
"""

__all__ = ["analyze_scores_entry_point"]

from pathlib                                    import Path
from typing                                     import Any, Dict, List, Set, Union

from gradus.commands.analyze_scores.__args__    import AnalyzeScoresConfig
from gradus.registration                        import register_command

@register_command(
    id =        "analyze-scores",
    config =    AnalyzeScoresConfig
)
def analyze_scores_entry_point(
    dataset_id:     str,
    output_path:    Union[str, Path] =  "analysis/datasets",
    *args,
    **kwargs
) -> Path:
    """# Analyze Metric Distributions for a Score Dataset.

    ## Args:
        * dataset_id    (str):          Identifier of dataset whose metric distributions are being calculated.
        * output_path   (str | Path):   Directory under which results will be written. Defaults to 
                                        "./analysis/datasets/".

    ## Returns:
        * Path: Path at which CSV results file was saved.
    """
    from logging            import Logger

    from pandas             import DataFrame, read_csv, Series

    from gradus.utilities   import get_logger

    # Initialize logger.
    logger:         Logger =    get_logger("distro-analysis")

    # Resolve paths.
    output_dir:     Path =      Path(output_path) / dataset_id
    scores_path:    Path =      output_dir / "metric-scores.csv"

    # If metrics have not been recorded...
    if not scores_path.exists():

        # Offer to initiate that process.
        if input(
            "Metric scores not yet calculated. Commence dataset scoring now? [Y/n] "
        ).strip().lower() not in ["n", "no"]:
            
            # Import process.
            from gradus.commands.score_dataset.__main__ import score_dataset_entry_point

            # Score dataset.
            score_dataset_entry_point(
                dataset_id =    dataset_id,
                output_path =   output_dir
            )

        # Otherwise, simply exit.
        else: logger.info(f"Aborting operation..."); exit()

    # Read metric scores into DataFrame.
    metric_scores:  DataFrame = read_csv(scores_path)
    
    # Get list of unique classes.
    classes:        Set[str] =  metric_scores["class"].unique()
    
    # For each metric discovered in file...
    for metric in [m for m in metric_scores.columns if m not in {"index", "class"}]:

        # Initialize dataframe rows.
        rows:   List[Dict[str, Any]] =  []
    
        # Extract metric series.
        series: Series =                metric_scores[metric]

        # Write global distribution.
        rows.append({
            "scope":    "global",
            "count":    int(series.count()),
            "mean":     float(series.mean()),
            "maximum":  float(series.max()),
            "median":   float(series.median()),
            "minimum":  float(series.min()),
            "std":      float(series.std()),
            "p25":      float(series.quantile(0.25)),
            "p50":      float(series.quantile(0.50)),
            "p75":      float(series.quantile(0.75))
        })

        # For each class in dataset...
        for cls in classes:

            # Extract filtered series.
            series: Series =    metric_scores.loc[metric_scores["class"] == cls, metric]

            # Write class distribution.
            rows.append({
                "scope":    cls,
                "count":    int(series.count()),
                "mean":     float(series.mean()),
                "maximum":  float(series.max()),
                "median":   float(series.median()),
                "minimum":  float(series.min()),
                "std":      float(series.std()),
                "p25":      float(series.quantile(0.25)),
                "p50":      float(series.quantile(0.50)),
                "p75":      float(series.quantile(0.75))
            })

        # Write distributions to file.
        DataFrame(rows).to_csv(output_dir / f"{metric}-distributions.csv", index = False)

    # Log file location.
    logger.info(f"Distribution calculations saved to {output_dir}")