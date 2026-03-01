"""# gradus.commands.score_sample.main

Main process entry point for score-sample command.
"""

__all__ = ["score_sample_entry_point"]

from pathlib                                import Path
from typing                                 import Any, Callable, Dict, List, Optional, Union

from gradus.commands.score_sample.__args__  import ScoreSampleConfig
from gradus.metrics                         import *
from gradus.registration                    import register_command

# Define possible metric mapping.
METRIC_MAP: Dict[str, Callable] =   {
                                        "color-variance":       color_variance,
                                        "compression-ratio":    compression_ratio,
                                        "edge-density":         edge_density,
                                        "spatial-frequency":    spatial_frequency,
                                        "time-to-convergence":  time_to_convergence,
                                        "time-to-saturation":   time_to_saturation,
                                        "wavelet-energy":       wavelet_energy,
                                        "wavelet-entropy":      wavelet_entropy,
                                    }

@register_command(
    id =        "score-sample",
    config =    ScoreSampleConfig
)
def score_sample_entry_point(
    sample_path:    Union[str, Path],
    metrics:        List[str] =                     ["all"],
    save_to:        Optional[Union[str, Path]] =    None
) -> Dict[str, Any]:
    """# Score Sample.

    ## Args:
        * sample_path   (str | Path):   Path from which sample can be loaded.
        * metrics       (List[str]):    Metric(s) being calculated for sample. Defaults to all.
        * save_to:      (str | Path):   Path at which complexity metrics report will be saved.

    ## Returns:
        * Dict[str, Any]:   Sample complexity metrics.
    """
    from json                               import dump
    from logging                            import Logger

    from PIL                                import Image
    from torch                              import Tensor
    from torchvision.transforms.functional  import to_tensor

    from gradus.utilities                   import get_logger

    # Initialize logger.
    logger: Logger =    get_logger("sample-scoring")

    try:# Load image as tensor.
        image:  Tensor =    to_tensor(Image.open(sample_path))

        # Resolve scheduled metrics.
        scheduled:  Dict[str, Callable] =   METRIC_MAP if "all" in metrics else {
                                                k: v 
                                                for k, v 
                                                in METRIC_MAP.items() 
                                                if k in metrics
                                            }
        
        # Calculate metrics.
        results:    Dict[str, Any] =        {name: fn(image) for name, fn in scheduled.items()}

        # If save path is provided...
        if save_to is not None:
            
            # Save metric mapping to file.
            with open(save_to, "w") as f: dump(results, f, indent = 2, default = str)

        # Return results.
        return results

    # Report file not found error.
    except FileNotFoundError: logger.error(f"Sample path does not exist: {sample_path}"); raise

from gradus.utilities import set_seed
from os import listdir
for seed in [1, 2, 3]:
    print(f"SEEDING {seed}")
    set_seed(seed)
    for sample in listdir("test_samples"):
        print(f"{sample} metrics: {score_sample_entry_point(f"test_samples/{sample}")}")