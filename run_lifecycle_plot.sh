#!/bin/bash
#==================================================================================================#
# LIFECYCLE CURRICULUM PILOT  (sample level, clean CIFAR-100, ResNet-18, 100 epochs)                #
#==================================================================================================#
# Each sample: introduced when learnable -> pruned when learned (3 correct passes in a row) ->      #
# re-checked on a doubling (spaced-repetition) interval -> revived if forgotten.                   #
#   1. full lifecycle, then prune / linear / cosine / inv-cosine (accuracy criterion) matched to    #
#      its savings                                                                                 #
#   2. ablations: no pacing stage (all data from the start), no revival, neither                   #
# Reference: sample-level full data (already recorded). 8 runs per seed (~4 h).                    #
#==================================================================================================#

set -e
H="python scratch_batch_pacing.py --unit sample"

stage() {       # seed
    $H --schedule full --seed $1
    $H --schedule lifecycle --seed $1
    f=$($H --schedule lifecycle --seed $1 --matched-fmin)
    echo "Lifecycle seed $1: matched f_min = $f"
    if [ "$f" = "0.000" ]; then
        echo "  -> lifecycle saved 50% or more; fixed shapes cannot match it (skipped). Send me this log line."
    else
        for s in prune linear cosine inv-cosine; do
            $H --schedule $s --criterion accuracy --f-min $f --seed $1
        done
    fi
    $H --schedule lifecycle --life-start 1 --seed $1
    $H --schedule lifecycle --no-revive --seed $1
    $H --schedule lifecycle --life-start 1 --no-revive --seed $1
}

stage 1;    $H --summarize
stage 2;    $H --summarize
