#!/bin/bash
#==================================================================================================#
# LIFECYCLE CURRICULUM: STRICTER "LEARNED" RULE  (sample level, clean CIFAR-100, ResNet-18)          #
#==================================================================================================#
# The pilot (3 correct passes = learned) saved 65-73%, beyond the 25-50% range where fixed shapes   #
# can be matched. Here a sample must be correct on K passes in a row (K = 6, 10).                  #
# Per K and seed:                                                                                  #
#   full lifecycle  vs  its ablation without pacing & revival   (does the lifecycle add anything?) #
#   prune, linear, cosine, inv-cosine (accuracy criterion) at the full lifecycle's savings          #
# Up to 12 runs per seed (~6 h). Safe to Ctrl-C and re-run to resume.                              #
#==================================================================================================#

set -e
H="python scratch_batch_pacing.py --unit sample"

stage() {       # seed
    for k in 6 10; do
        $H --schedule lifecycle --life-k $k --seed $1
        $H --schedule lifecycle --life-k $k --life-start 1 --no-revive --seed $1
        f=$($H --schedule lifecycle --life-k $k --seed $1 --matched-fmin)
        echo "Lifecycle K=$k seed $1: matched f_min = $f"
        if [ "$f" = "0.000" ]; then
            echo "  -> still 50% or more savings; fixed shapes skipped."
        else
            for s in prune linear cosine inv-cosine; do
                $H --schedule $s --criterion accuracy --f-min $f --seed $1
            done
        fi
    done
}

stage 1;    $H --summarize
stage 2;    $H --summarize
