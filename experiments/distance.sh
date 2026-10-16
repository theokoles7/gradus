#!/bin/bash

#==================================================================================================#
# DISTANCE-BASED CURRICULA EXPERIMENTS                                                             #
#==================================================================================================#
# Total experiments: 768                                                                           #
#                                                                                                  #
# This shell script automates a comprehensive set of weighted curriculum learning experiments by   #
# exhaustively evaluating all combinations of experimental parameters.                             #
#==================================================================================================#

for scope in batch-wise holistic; do

    for dataset in cifar-10 cifar-100; do

        # for model in resnet-18 resnet-34 resnet-50 resnet-101 vgg-11 vgg-13 vgg-16 vgg-19; do
        for model in resnet-18; do

            for metric in color-variance compression-ratio edge-density spatial-frequency wavelet-energy wavelet-entropy saturation-time convergence-time; do

                for seed in 1 2 3; do

                    time gradus train --epochs 100 --seed $seed $model $dataset --rank distance-from-mean --metric $metric --scope $scope

                done

            done

        done
    
    done

done