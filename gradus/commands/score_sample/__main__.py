"""# gradus.commands.score_sample.main

Main process entry point for score-sample command.
"""

__all__ = ["score_sample_entry_point"]

from pathlib                                import Path
from typing                                 import Any, Dict, List, Optional, Union

from gradus.commands.score_sample.__args__  import ScoreSampleConfig
from gradus.registration                    import register_command

@register_command(
    id =        "score-sample",
    config =    ScoreSampleConfig
)
def score_sample_entry_point(
    sample_path:    Union[str, Path],
    metrics:        List[str] =                     ["all"],
    save_to:        Optional[Union[str, Path]] =    None,
    device:         str =                           "auto",
    seed:           int =                           1,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """# Score Sample.

    ## Args:
        * sample_path   (str | Path):   Path from which sample can be loaded.
        * metrics       (List[str]):    Metric(s) to compute. Defaults to all registered metrics.
        * save_to       (str | Path):   Path at which results will be saved (JSON). Defaults to 
                                        None.
        * device        (str):          Torch computation device. Defaults to "auto".
        * seed          (int):          Random number generation seed. Defaults to 1.

    ## Returns:
        * Dict[str, Any]:   Mapping of metric IDs to their computed values.
    """
    from json                               import dump
    from logging                            import Logger

    from PIL                                import Image
    from torch                              import Tensor
    from torchvision.transforms.functional  import to_tensor

    from gradus.registration                import METRIC_REGISTRY
    from gradus.utilities                   import determine_device, get_logger, set_seed

    # Initialize logger.
    logger: Logger =    get_logger("sample-scoring")

    # Set device & seed.
    determine_device(device); set_seed(seed)

    try:# Load image as tensor.
        sample:         Tensor =            to_tensor(Image.open(sample_path))

        # Normalize metrics argument (argparse nargs="+" always gives a list; direct calls may not).
        metrics_list:   List[str] =         metrics if isinstance(metrics, list) else [metrics]

        # Resolve which metrics to run.
        scheduled:      List[str] =         METRIC_REGISTRY.list_entries()  \
                                            if "all" in metrics_list        \
                                            else [m for m in metrics_list if m in METRIC_REGISTRY]

        # For any unrecognized metrics...
        for uid in [m for m in metrics_list if m != "all" and m not in METRIC_REGISTRY]:

            # Warn that it will not be computed.
            logger.warning(f"Metric not registered, skipping: {uid}")

        # Compute metrics, isolating failures so one bad metric doesn't abort the run.

        # Initialize calculations map.
        results:        Dict[str, Any] =    {}

        # For each scheduled metric...
        for metric_id in scheduled:

            # Calculate metric for sample.
            try: results[metric_id] = METRIC_REGISTRY.get_entry(metric_id).fn(sample)

            # Report calculation failures.
            except Exception as e: results[metric_id] = str(e)

        # Log results.
        for metric_id, value in results.items(): logger.info(f"""{f"{metric_id}:":20}{value}""")

        # If a save path is provided...
        if save_to is not None:

            # Open file for writing.
            with open(save_to, "w") as f:

                # Write results to file.
                dump(results, f, indent = 2, default = str)

            # Communicate save location.
            logger.info(f"Results saved to: {Path(save_to).absolute()}")

        # Provide results to outer process.
        return results

    # Report file not found error.
    except FileNotFoundError: logger.error(f"Sample path does not exist: {sample_path}"); raise