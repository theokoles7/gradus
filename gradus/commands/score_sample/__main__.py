# """# gradus.commands.score_sample.main

# Main process entry point for score-sample command.
# """

# __all__ = ["score_sample_entry_point"]

# from pathlib                                import Path
# from typing                                 import Any, Callable, Dict, List, Optional, Union

# from gradus.commands.score_sample.__args__  import ScoreSampleConfig
# from gradus.registration                    import register_command

# @register_command(
#     id =        "score-sample",
#     config =    ScoreSampleConfig
# )
# def score_sample_entry_point(
#     sample_path:    Union[str, Path],
#     metrics:        List[str] =                     ["all"],
#     save_to:        Optional[Union[str, Path]] =    None,
#     *args,
#     **kwargs
# ) -> Dict[str, Any]:
#     """# Score Sample.

#     ## Args:
#         * sample_path   (str | Path):   Path from which sample can be loaded.
#         * metrics       (List[str]):    Metric(s) being calculated for sample. Defaults to all.
#         * save_to:      (str | Path):   Path at which complexity metrics report will be saved.

#     ## Returns:
#         * Dict[str, Any]:   Sample complexity metrics.
#     """
#     from json                               import dump
#     from logging                            import Logger

#     from PIL                                import Image
#     from torch                              import Tensor
#     from torchvision.transforms.functional  import to_tensor

#     from gradus.registration                import METRIC_REGISTRY
#     from gradus.utilities                   import get_logger

#     # Initialize logger.
#     logger: Logger =    get_logger("sample-scoring")

#     try:# Load image as tensor.
#         image:  Tensor =                    to_tensor(Image.open(sample_path))

#         # Resolve scheduled metrics.
#         scheduled:  Dict[str, Callable] =   METRIC_REGISTRY.list_entries()                  \
#                                             if "all" in metrics else                        \
#                                             [m for m in metrics if m in METRIC_REGISTRY]
        
#         # Calculate metrics.
#         results:    Dict[str, Any] =        {m: m.fn(image) for m, fn in scheduled.items()}

#         # If save path is provided...
#         if save_to is not None:
            
#             # Save metric mapping to file.
#             with open(save_to, "w") as f: dump(results, f, indent = 2, default = str)

#         # Log metric calculations.
#         for m, v in results.items(): logger.info(f"{m}:\t{v}")

#         # Return results.
#         return results

#     # Report file not found error.
#     except FileNotFoundError: logger.error(f"Sample path does not exist: {sample_path}"); raise

# from gradus.utilities import set_seed
# from os import listdir
# for seed in [1, 2, 3]:
#     print(f"SEEDING {seed}")
#     set_seed(seed)
#     for sample in listdir("test_samples"):
#         print(f"{sample} metrics: {score_sample_entry_point(f"test_samples/{sample}")}")

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
    *args,
    **kwargs
) -> Dict[str, Any]:
    """# Score Sample.

    ## Args:
        * sample_path   (str | Path):   Path from which sample can be loaded.
        * metrics       (List[str]):    Metric(s) to compute. Defaults to all registered metrics.
        * save_to       (str | Path):   Path at which results will be saved (JSON). Defaults to 
                                        None.

    ## Returns:
        * Dict[str, Any]:   Mapping of metric IDs to their computed values.
    """
    from json                               import dump
    from logging                            import Logger

    from PIL                                import Image
    from torch                              import Tensor
    from torchvision.transforms.functional  import to_tensor

    from gradus.registration                import METRIC_REGISTRY
    from gradus.utilities                   import get_logger

    # Initialize logger.
    logger: Logger =    get_logger("sample-scoring")

    try:
        # Load image as tensor.
        sample: Tensor =    to_tensor(Image.open(sample_path))

        # Normalize metrics argument (argparse nargs="+" always gives a list; direct calls may not).
        metrics_list:   List[str] = metrics if isinstance(metrics, list) else [metrics]

        # Resolve which metrics to run.
        if "all" in metrics_list:
            metric_ids: List[str] = METRIC_REGISTRY.list_entries()
        else:
            metric_ids: List[str] = [m for m in metrics_list if m in METRIC_REGISTRY]

        # Warn about any unrecognised metric IDs.
        for uid in [m for m in metrics_list if m != "all" and m not in METRIC_REGISTRY]:
            logger.warning(f"Metric not registered, skipping: {uid}")

        # Compute metrics, isolating failures so one bad metric doesn't abort the run.
        results: Dict[str, Any] = {}
        for metric_id in metric_ids:
            try:
                results[metric_id] = METRIC_REGISTRY.get_entry(metric_id).fn(sample)
            except Exception as e:
                logger.error(f"Failed to compute '{metric_id}': {e}")
                results[metric_id] = None

        # Log results.
        for metric_id, value in results.items():
            logger.info(f"{metric_id}: {value}")

        # Optionally save results to JSON.
        if save_to is not None:
            with open(save_to, "w") as f:
                dump(results, f, indent = 2, default = str)
            logger.info(f"Results saved to: {Path(save_to).absolute()}")

        # Return results.
        return results

    # Report file not found error.
    except FileNotFoundError:
        logger.error(f"Sample path does not exist: {sample_path}"); raise