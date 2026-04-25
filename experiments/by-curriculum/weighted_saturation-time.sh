#!/bin/bash

# Run all experiments with CIFAR-10 dataset.

for model in resnet-18 resnet-34 resnet-50 resnet-101; do

    for dataset in cifar-10 cifar-100; do

        time gradus train $model $dataset --rank weighted --metric saturation-time
    
    done

done