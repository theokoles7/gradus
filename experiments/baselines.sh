#!/bin/bash

# +================================================================================+
# | Run all baseline experiments.                                                  |
# | -----------------------------                                                  |
# | Baseline experiments include:                                                  |
# | 1. Training a model on the original dataset, as it is loaded from torchvision. |
# | 2. Training a model on the dataset using random shuffling.                     |
# +================================================================================+

# CONFIGURATION ====================================================================================
EPOCHS=100

# EXPERIMENT =======================================================================================

# For each model implemented...
for model in resnet-18 resnet-34 resnet-50 resnet-101; do

    # For each dataset...
    for dataset in  mnist cifar-10 cifar-100; do

        # Run true baseline.
        time gradus train --epochs $EPOCHS $model $dataset

        # Run shuffled baseline.
        time gradus train --epochs $EPOCHS $model $dataset --shuffle

    done

done