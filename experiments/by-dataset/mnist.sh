#!/bin/bash

# Run all experiments with MNIST dataset.

for model in resnet-18 resnet-34 resnet-50 resnet-101; do

    # Baseline
    time gradus train $model mnist

    # Shuffled
    time gradus train $model mnist --shuffle

    # Weighted by Saturation Time
    time gradus train $model mnist --rank weighted --metric saturation-time

    # Weighted by Saturation Time (Linear Schedule)
    time gradus train $model mnist --rank weighted --metric saturation-time --schedule linear

    # Weighted by Saturation Time (Adaptive Schedule)
    time gradus train $model mnist --rank weighted --metric saturation-time --schedule adaptive

done