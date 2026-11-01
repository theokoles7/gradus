#!/bin/bash
#==================================================================================================#
# BATCH-LEVEL PACING, STAGE 2  (clean CIFAR-100, ResNet-18, 100 epochs, fixed batch partition)      #
#==================================================================================================#
# Part A - vs. B-PAS (pruning-only reference, paper defaults delta 1e-6 -> 5e-5):                  #
#   B-PAS runs first; linear pacing, sine, cosine and inv-cosine (pace in, then prune) then run    #
#   with B-PAS's criterion at B-PAS's realized savings.                                            #
# Part B - inv-cosine for the stage-1 criteria (accuracy, gradient, activation) at 25% savings,    #
#   compared with the stage-1 prune / linear / sine / cosine runs.                                 #
# 8 runs per seed (~4 h). Stage-1 runs are reused. Safe to Ctrl-C and re-run to resume.            #
#==================================================================================================#

set -e
H="python scratch_batch_pacing.py"

stage() {       # seed
    $H --schedule full --seed $1
    $H --schedule bpas --criterion bpas --seed $1
    f=$($H --schedule bpas --criterion bpas --seed $1 --matched-fmin)
    echo "B-PAS seed $1: matched f_min = $f"
    for s in linear sine cosine inv-cosine; do
        $H --schedule $s --criterion bpas --f-min $f --seed $1
    done
    for c in accuracy gradient activation; do
        $H --schedule inv-cosine --criterion $c --seed $1
    done
}

stage 1;    $H --summarize
stage 2;    $H --summarize
