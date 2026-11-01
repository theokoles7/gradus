#!/bin/bash
#==================================================================================================#
# BI-DIRECTIONAL PACING, STAGE 1 (clean data; CIFAR-100, ResNet-18, 100 epochs)                    #
#==================================================================================================#
# Schedules (all between 50% and 100% of the data, ~75% compute on the epoch clock):               #
#   full    all data                                                                               #
#   prune   pruning only            1.0 -> 0.5                                                     #
#   pace    pacing only             0.5 -> 1.0  (incremental introduction)                         #
#   sine    bi-directional          0.75 -> 1.0 -> 0.5 -> 0.75                                     #
#   cosine  bi-directional          1.0 -> 0.5 -> 1.0                                              #
#                                                                                                  #
# Part A: random selection; fixed (epoch) clock vs accuracy-paced clock (train accuracy).          #
# Part B: the same four schedules, accuracy-paced, with dynamic (EL2N, loss) & static              #
#         (spatial frequency, compression ratio) metrics, each with its random replay control.    #
# Every method also gets its compute-matched short schedule (shared when lengths coincide).       #
#                                                                                                  #
# Seed 1: Part A ~5 h, Part B ~14 h. Then seed 2 of both. Safe to Ctrl-C and re-run to resume.     #
#==================================================================================================#

set -e

PACED="--clock signal --signal train_acc"

method() {      # args...: run a method, then its controls (replay only for non-random metrics)
    python scratch_bidirectional.py "$@"
    python scratch_bidirectional.py "$@" --control short
    case "$*" in *"--metric random"*) ;; *) python scratch_bidirectional.py "$@" --control replay ;; esac
}

part_a() {      # seed
    python scratch_bidirectional.py --schedule full --seed $1
    for s in prune pace sine cosine; do
        method --schedule $s --metric random --clock epoch --seed $1
        method --schedule $s --metric random $PACED --seed $1
    done
}

part_b() {      # seed
    for m in el2n loss static:spatial-frequency static:compression-ratio; do
        for s in prune pace sine cosine; do
            method --schedule $s --metric $m $PACED --seed $1
        done
    done
}

part_a 1;   python scratch_bidirectional.py --summarize
part_b 1;   python scratch_bidirectional.py --summarize
part_a 2;   part_b 2;   python scratch_bidirectional.py --summarize