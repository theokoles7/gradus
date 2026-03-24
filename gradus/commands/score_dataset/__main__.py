"""# gradus.commands.score_dataset.main

Main process entry point for score-dataset command.
"""

__all__ = ["score_dataset_entry_point"]

from pathlib                                    import Path
from typing                                     import Any, Dict, List, Set, Union

from gradus.commands.score_dataset.__args__     import ScoreDatasetConfig
from gradus.registration                        import register_command

@register_command(
    id =        "score-dataset",
    config =    ScoreDatasetConfig
)
def score_dataset_entry_point(
    dataset_id:     str,
    metrics:        List[str] =         ["all"],
    output_path:    Union[str, Path] =  ".cache/datasets",
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
                                        "./,cache/datasets/".
        * device        (str):          Torch computation device. Defaults to "auto".
        * seed          (int):          Random number generation seed. Defaults to 1.

    ## Returns:
        * Path: Path at which CSV results file was saved.
    """
    from logging                import Logger

    from torch                  import Tensor
    from tqdm                   import tqdm

    from gradus.artifacts       import DatasetMetrics
    from gradus.datasets        import Dataset
    from gradus.registration    import DATASET_REGISTRY, METRIC_REGISTRY
    from gradus.utilities       import determine_device, get_logger, set_seed

    # Set device & seed.
    determine_device(device); set_seed(seed)

    # Initialize logger.
    logger:         Logger =            get_logger("dataset-scoring")

    # Load dataset.
    dataset:        Dataset =           DATASET_REGISTRY.load_dataset(dataset_id, **kwargs)

    # Load dataset metrics.
    scores:         DatasetMetrics =    DatasetMetrics(
                                            dataset_id =    dataset_id,
                                            num_samples =   len(dataset.train_data),
                                            seed =          seed,
                                            scores_path =   output_path
                                        )

    # Normalize metrics argument.
    metrics_list:   List[str] =         metrics if isinstance(metrics, list) else [metrics]

    # Resolve which metrics to run.
    scheduled:      List[str] =         METRIC_REGISTRY.list_entries()  \
                                        if "all" in metrics_list        \
                                        else [m for m in metrics_list if m in METRIC_REGISTRY]

    # For any unrecognized metrics...
    for uid in [m for m in metrics_list if m != "all" and m not in METRIC_REGISTRY]:

        # Warn that it will not be computed.
        logger.warning(f"Metric not registered, skipping: {uid}")

    # Determine set of unscored sample indices per metric.
    unscored:       Dict[str, Set] =    {
                                            metric_id:  scores.get_unscored(metric_id)
                                            for metric_id in scheduled
                                        }
    
    # If all scores have already been computed...
    if not any(unscored.values()):
        
        # Log condition & exit.
        logger.info(f"{dataset_id}, seed {seed} already scored"); return scores.scores_path

    # Log process initiation.
    logger.info(f"Scoring {dataset_id.upper()} ({len(dataset.train_data)} samples; metrics: {scheduled})")

    # For each sample...
    for i in tqdm(
        range(len(dataset.train_data)),
        desc =  f"Scoring {dataset_id.upper()}",
        unit =  "sample(s)"
    ):            
    #     # Unpack sample & label.
    #     sample: Tensor =            dataset.train_data[i][0]
    #     label:  Any =               dataset.train_data[i][1]

    #     # Initialize row with index & label.
    #     row:    Dict[str, Any] =    {"index": i, "class": dataset.classes[label]}

    #     # For each scheduled metric...
    #     for metric_id in scheduled:

    #         # If metric has already been computed for sample...
    #         if scores.get(index = i, metric = )

    #         # Calculate metric for sample.
    #         try: row[metric_id] = METRIC_REGISTRY.get_entry(metric_id).fn(sample)

    #         # Record failed caluclations as NAN.
    #         except Exception as e: row[metric_id] = float("nan"); raise

        # Determine which metrics still need computing for this sample.
        to_compute: List[str] = [m for m in scheduled if i in unscored[m]]
 
        # If nothing to compute for this sample, move on.
        if not to_compute: continue
 
        # Unpack sample & label.
        sample: Tensor =            dataset.train_data[i][0]
        label:  Any =               dataset.train_data[i][1]
 
        # Compute each outstanding metric for this sample.
        row:    Dict[str, Any] =    {}
 
        for metric_id in to_compute:
            try:    row[metric_id] = METRIC_REGISTRY.get_entry(metric_id).fn(sample)
            except Exception as e:
                row[metric_id] = float("nan")
                logger.warning(f"Metric '{metric_id}' failed for sample {i}: {e}")
 
        # Record row.
        scores.record_row(index = i, label = dataset.classes[label], scores = row)
 
    # Provide results to outer process.
    return scores.save()