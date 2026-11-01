#!/bin/bash
#==================================================================================================#
# STAGE 9a: CALIBRATION RE-RUN WITH PER-SAMPLE SNAPSHOTS (early-epoch only; no results rows)      #
#==================================================================================================#
# Full-data training under the normal 100-epoch schedule, stopped after epoch 15. Logs per-epoch   #
# noise-rate estimates (negative margin, EL2N > 1, GMM) and flipped-label AUCs. Two datasets and   #
# a second architecture test whether one estimator->noise-rate mapping is portable.                #
# Saves per-sample state at epochs 5, 10, 15 for offline estimator design (scratch_estimator_lab).#
# 31 runs, each ~15% of a full run (~3 h total). Safe to stop and re-run to resume.                #
#==================================================================================================#

set -e

CAL="--epochs 100 --prune-fraction 0 --stop-after 15"

for dataset in cifar-100 cifar-10; do
    for seed in 1 2; do
        S="--network resnet-18 --dataset $dataset --seed $seed $CAL"
        for noise in 0 0.1 0.2 0.3 0.4; do
            python scratch_prune_ablation.py $S --label-noise $noise
        done
        for noise in 0.2 0.4; do
            python scratch_prune_ablation.py $S --label-noise $noise --noise-type pairflip
        done
    done
done

# Architecture check: ResNet-50 on CIFAR-100, one seed.
for noise in 0 0.2 0.4; do
    python scratch_prune_ablation.py --network resnet-50 --dataset cifar-100 --seed 1 $CAL --label-noise $noise
done

python scratch_calibration_summary.py
python scratch_estimator_lab.py