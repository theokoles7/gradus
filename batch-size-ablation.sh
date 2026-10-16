#!/bin/bash

#==================================================================================================#
# BATCH-SIZE ABLATION                                                                              #
#==================================================================================================#
# Compares shuffled training vs holistic ascending curriculum at batch sizes 8, 16, 32, and 64.   #
# Total experiments: 24 = 4 batch sizes x 2 conditions x 3 seeds                                  #
#==================================================================================================#

dataset=cifar-100
model=resnet-18
metric=wavelet-energy

for batch_size in 8 16 32 64; do

    for seed in 1 2 3; do

        # Shuffled baseline
        time gradus train \
            --epochs 100 \
            --seed $seed \
            $model $dataset \
            --batch-size $batch_size

        # Holistic ascending curriculum
        time gradus train \
            --epochs 100 \
            --seed $seed \
            $model $dataset \
            --batch-size $batch_size \
            --rank ascending \
            --metric $metric \
            --shuffle

    done

done