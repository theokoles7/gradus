#!/bin/bash
#==================================================================================================#
# SAMPLE-LEVEL PACING, STAGE 1  (clean CIFAR-100, ResNet-18, 100 epochs, --unit sample)             #
#==================================================================================================#
# The batch-level matrix (stages 1 + 2), repeated with single samples as the unit:                 #
#   per criterion (accuracy, gradient, activation):                                                #
#       prune, linear, sine, cosine, inv-cosine          at 25% savings                            #
#       dynamic, then prune & linear at the dynamic run's savings                                  #
#   B-PAS applied per sample, then linear, sine, cosine, inv-cosine at B-PAS's savings             #
# Reference: full-data training (standard reshuffled batches).                                     #
# 30 runs per seed (~15 h). Safe to Ctrl-C and re-run to resume.                                   #
#==================================================================================================#

set -e
H="python scratch_batch_pacing.py --unit sample"

stage() {       # seed
    $H --schedule full --seed $1
    for c in accuracy gradient activation; do
        for s in prune linear sine cosine inv-cosine; do
            $H --schedule $s --criterion $c --seed $1
        done
        $H --schedule dynamic --criterion $c --seed $1
        f=$($H --schedule dynamic --criterion $c --seed $1 --matched-fmin)
        $H --schedule prune  --criterion $c --f-min $f --seed $1
        $H --schedule linear --criterion $c --f-min $f --seed $1
    done
    $H --schedule bpas --criterion bpas --seed $1
    f=$($H --schedule bpas --criterion bpas --seed $1 --matched-fmin)
    echo "B-PAS (sample level) seed $1: matched f_min = $f"
    for s in linear sine cosine inv-cosine; do
        $H --schedule $s --criterion bpas --f-min $f --seed $1
    done
}

stage 1;    $H --summarize
stage 2;    $H --summarize
