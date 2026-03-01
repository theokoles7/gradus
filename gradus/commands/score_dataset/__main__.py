"""# gradus.commands.analyze_dataset.main

Main process entry point for score-dataset command.
"""

__all__ = ["analyze_dataset_entry_point"]

from typing                                     import Dict

from gradus.commands.score_dataset.__args__     import ScoreDatasetConfig
from gradus.metrics                             import *
from gradus.registration                        import register_command

@register_command(
    id =        "score-dataset",
    config =    ScoreDatasetConfig
)
def score_dataset_entry_point(
    dataset_id:     str,
    output_path:    str =   "analyses/datasets",
    *args,
    **kwargs
) -> Dict[int, Dict]:
    """# Score Dataset and Compute Image Complexity Metrics.

    ## Args:
        * dataset_id    (str):  Identifier of dataset being analyzed.
        * output_path   (str):  Path at which dataset analysis results will be written. Defaults 
                                to "./analyses/datasets/".
    """
    from json                   import dump
    from logging                import Logger
    from pathlib                import Path
    from typing                 import List

    from matplotlib.pyplot      import subplots, xticks
    from pandas                 import DataFrame, melt
    from seaborn                import barplot
    from torch                  import Tensor
    from tqdm                   import tqdm

    from gradus.datasets        import Dataset
    from gradus.registration    import DATASET_REGISTRY
    from gradus.utilities       import get_logger

    # Form output path.
    output_path:    Path =              Path(output_path) / dataset_id

    # Initialize logger.
    logger:         Logger =            get_logger("dataset-analysis-process")

    # Load dataset.
    dataset:        Dataset =           DATASET_REGISTRY.load_dataset(dataset_id)

    # Initialize analysis mapping.
    analysis:       Dict[int, Dict] =   {i: {} for i in range(len(dataset.train_data))}

    # For each sample in the dataset...
    for i in tqdm(
        range(len(dataset.train_data)),
        desc = f"Analyzing {dataset_id.upper()}",
        unit = "sample(s)"
    ):

        # Get sample.
        sample: Tensor =                            dataset.train_data[i][0]

        # Take note of the class of the sample.
        analysis[i]["class"] =                      dataset.train_data[i][1]

        # Compute and store image complexity metrics.
        analysis[i]["color_variance"] =             color_variance(image =    sample[0])
        analysis[i]["compression_ratio"] =          compression_ratio(image = sample[0])
        analysis[i]["edge_density"] =               edge_density(image =      sample[0])
        rf, cf, of =                                spatial_frequency(image = sample[0])
        analysis[i]["wavelet_energy"] =             wavelet_energy(image =    sample[0]).total_energy
        analysis[i]["wavelet_entropy"] =            wavelet_entropy(image =   sample[0]).entropy
        
        # Unpack the tuple (RF, CF, OF).
        analysis[i]["spatial_frequency_row"] =      rf
        analysis[i]["spatial_frequency_column"] =   cf
        analysis[i]["spatial_frequency_overall"] =  of

    # Ensure outupt path's parent directory exists.
    output_path.mkdir(parents = True, exist_ok = True)

    # Save analysis to JSON file.
    with open(output_path / f"{dataset_id}-complexity-metrics.json", "w") as f: dump(obj = analysis, fp = f, indent = 2)

    # Log save location.
    logger.info(f"Metric scores saved to: {output_path.absolute()}/{dataset_id}-complexity-metrics.json")

    # Convert analysis to DataFrame.
    df:             DataFrame =         DataFrame.from_dict(analysis, orient = "index")

    # Save DataFrame to CSV file.
    df.to_csv(output_path / f"{dataset_id}-complexity-metrics.csv")

    # Plot distributions of standard metrics by class.
    metrics:        List[str] =         ["color_variance", "compression_ratio", "edge_density", "wavelet_energy", "wavelet_entropy"]

    # For each metric analyzed...
    for metric in metrics:

        # Create a wide figure to accommodate all 100 classes.
        fig, ax = subplots(figsize = (20, 6))
        
        # Use barplot to show mean with error bars per class.
        barplot(x = "class", y = metric, data = df, ax = ax, errorbar = "sd")
        
        ax.set_title(f"{metric.replace('_', ' ').title()} by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Rotate x-axis labels for readability.
        xticks(rotation = 90, fontsize = 8)
        fig.tight_layout()
        
        # Save figure.
        fig.savefig(output_path / f"{dataset_id}-{metric}-by-class.png")

        # Log save location.
        logger.info(f"Saved {metric} by class plot to: {output_path.absolute()}/{dataset_id}-{metric}-by-class.png")

    # Melt spatial frequency columns for grouped visualization.
    df_melted:      DataFrame =         melt(
                                            df[["class"] + ["spatial_frequency_row", "spatial_frequency_column", "spatial_frequency_overall"]],
                                            id_vars =       ["class"],
                                            var_name =      "frequency_type",
                                            value_name =    "value"
                                        )

    # Rename frequency types for better readability.
    df_melted["frequency_type"] = df_melted["frequency_type"].str.replace("spatial_frequency_", "").str.title()

    # Plot all spatial frequency components in one wider plot.
    fig, ax = subplots(figsize = (20, 6))
    barplot(x = "class", y = "value", hue = "frequency_type", data = df_melted, ax = ax, errorbar = "sd")
    ax.set_title("Spatial Frequency by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")
    xticks(rotation = 90, fontsize = 8)
    fig.tight_layout()

    # Save figure.
    fig.savefig(output_path / f"{dataset_id}-spatial-frequency-by-class.png")

    # Log save location.
    logger.info(f"Saved spatial frequency by class plot to: {output_path.absolute()}/{dataset_id}-spatial-frequency-by-class.png")

    # Return analysis.
    return analysis