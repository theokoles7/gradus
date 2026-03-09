"""# gradus.commands.score_dataset.main

Main process entry point for score-dataset command.
"""

__all__ = ["score_dataset_entry_point"]

from pathlib                                    import Path
from typing                                     import Any, Dict, List, Union

from gradus.commands.score_dataset.__args__     import ScoreDatasetConfig
from gradus.registration                        import register_command

@register_command(
    id =        "score-dataset",
    config =    ScoreDatasetConfig
)
def score_dataset_entry_point(
    dataset_id:     str,
    metrics:        List[str] =         ["all"],
    output_path:    Union[str, Path] =  "analysis/datasets",
    device:         str =               "auto",
    seed:           int =               1,
    *args,
    **kwargs
) -> Path:
    """# Score Dataset and Compute Image Complexity Metrics.

    ## Args:
        * dataset_id    (str):          Identifier of dataset being analyzed.
        * metrics       (List[str]):    Metric(s) to compute. Defaults to all registered metrics.
        * output_path   (str | Path):   Directory under which results will be written. Defaults to 
                                        "./analysis/datasets/".
        * device        (str):          Torch computation device. Defautls to "auto".
        * seed          (int):          Random number generation seed. Defaults to 1.

    ## Returns:
        * Path: Path at which CSV results file was saved.
    """
    from csv                    import DictWriter
    from logging                import Logger

    from torch                  import Tensor
    from tqdm                   import tqdm

    from gradus.datasets        import Dataset
    from gradus.registration    import DATASET_REGISTRY, METRIC_REGISTRY
    from gradus.utilities       import determine_device, get_logger, set_seed

    # Set device & seed.
    determine_device(device); set_seed(seed)

    # Initialize logger.
    logger:         Logger =    get_logger("dataset-scoring")

    # Form output path.
    output_dir:     Path =      Path(output_path) / dataset_id

    # Load dataset.
    dataset:        Dataset =   DATASET_REGISTRY.load_dataset(dataset_id, **kwargs)

    # Ensure path exists.
    output_dir.mkdir(parents = True, exist_ok = True)

    # Normalize metrics argument.
    metrics_list:   List[str] = metrics if isinstance(metrics, list) else [metrics]

    # Resolve which metrics to run.
    scheduled:      List[str] =         METRIC_REGISTRY.list_entries()  \
                                        if "all" in metrics_list        \
                                        else [m for m in metrics_list if m in METRIC_REGISTRY]

    # For any unrecognized metrics...
    for uid in [m for m in metrics_list if m != "all" and m not in METRIC_REGISTRY]:

        # Warn that it will not be computed.
        logger.warning(f"Metric not registered, skipping: {uid}")

    # Log process initiation.
    logger.info(f"Scoring {dataset_id.upper()} ({len(dataset.train_data)} samples; metrics: {scheduled})")

    # Determine CSV path.
    csv_path:       Path =              output_dir / f"{dataset_id}-complexity-metrics.csv"

    # Define CSV columns.
    columns:        List[str] =         ["index", "label"] + scheduled

    # Open CSV for incremental writing.
    with open(csv_path, "w", newline = "") as csv_file:

        # Initialize dictionary writer.
        writer = DictWriter(csv_file, fieldnames = columns); writer.writeheader()

        # For each sample...
        for i in tqdm(
            range(len(dataset.train_data)),
            desc =  f"Scoring {dataset_id.upper()}",
            unit =  "sample(s)"
        ):
            # Unpack sample & label.
            sample: Tensor =            dataset.train_data[i][0]
            label:  Any =               dataset.train_data[i][1]

            # Initialize row with index & label.
            row:    Dict[str, Any] =    {"index": i, "label": label}

            # Compute each metric, isolating failures.
            # For each scheduled metric...
            for metric_id in scheduled:

                # Calculate metric for sample.
                try: row[metric_id] = METRIC_REGISTRY.get_entry(metric_id).fn(sample)

                # Record failed caluclations as NAN.
                except Exception as e: row[metric_id] = float("nan")

            # Write row to file.
            writer.writerow(row)

    # Communicate save location.
    logger.info(f"Results saved to: {csv_path.absolute()}")

    # Provide results to outer process.
    return csv_path