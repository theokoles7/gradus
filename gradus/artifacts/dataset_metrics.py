"""# gradus.artifacts.dataset_metrics

Data structure implementation for storing dataset metric scores & statistics.
"""

class DatasetMetrics():
    """# Dataset Metrics
    
    Dataset metric scores & statistics.
    """

    def __init__(self,
        scores_path:    str =   ".cache/scores",
    ):
        """# Instantiate Dataset Metrics.
        
        ## Args:
            * scores_path   (str):  Path at which dataset scores will be written/managed. Defaults 
                                    to "./.cache/scores".
        """