#!/bin/bash

# Run all experiments with ImageNet dataset.

for model in resnet-18 resnet-34 resnet-50 resnet-101; do

    # Baseline
    time gradus train $model imagenet

    # Shuffled
    time gradus train $model imagenet --shuffle

    # Weighted by Saturation Time
    time gradus train $model imagenet --rank weighted --metric saturation-time

    # Weighted by Saturation Time (Linear Schedule)
    time gradus train $model imagenet --rank weighted --metric saturation-time --schedule linear

    # Weighted by Saturation Time (Adaptive Schedule)
    time gradus train $model imagenet --rank weighted --metric saturation-time --schedule adaptive

done