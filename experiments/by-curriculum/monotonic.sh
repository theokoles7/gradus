#!/bin/bash

# Run all experiments with CIFAR-10 dataset.

for model in resnet-18 resnet-34 resnet-50 resnet-101; do

    for dataset in cifar-10 cifar-100; do

        for metric in color-variance compression-ratio edge-density spatial-frequency wavelet-energy wavelet-entropy saturation-time convergence-time; do

            time gradus train --epochs 100 $model $dataset --rank ascending --metric $metric
            time gradus train --epochs 100 $model $dataset --rank descending --metric $metric

        done
    
    done

done