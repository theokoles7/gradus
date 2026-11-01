#!/bin/bash
#==================================================================================================#
# BI-DIRECTIONAL PACING, STAGE 2 (clean data; CIFAR-100, ResNet-18, 100 epochs)                    #
#==================================================================================================#
# Stage 1: dynamic retirement (EL2N / loss) beats random replay by ~0.9 but trails the compute-    #
# matched short schedule by ~0.4-0.5 at 75-80% compute; pacing from a subset costs 2+ points.      #
#                                                                                                  #
# Part A - lower budgets (where shortening training hurts more), EL2N, accuracy-paced clock:       #
#   prune 1 -> 0.3, prune 1 -> 0.1, cosine 1 <-> 0.3, cosine 1 <-> 0.1, cosine 0.6 <-> 0.2          #
# Part B - selection at the stage-1 budget (0.5), accuracy-paced clock:                            #
#   stochastic retirement (T = 1) for prune & cosine;                                              #
#   cosine reviving the HARDEST retired samples first (--intro-rank descending), EL2N & loss.      #
# Every arm: + replay control (random choices, same trajectory) + compute-matched short schedule.  #
# ~8 h per seed. Safe to Ctrl-C and re-run to resume; stage-1 runs are reused.                     #
#==================================================================================================#

set -e

PACED="--clock signal --signal train_acc"

method() {      # args...: run a method, then its short-schedule and replay controls
    python scratch_bidirectional.py "$@"
    python scratch_bidirectional.py "$@" --control short
    python scratch_bidirectional.py "$@" --control replay
}

part_a() {      # seed
    for f in 0.3 0.1; do
        method --schedule prune  --metric el2n $PACED --f-min $f --seed $1
        method --schedule cosine --metric el2n $PACED --f-min $f --seed $1
    done
    method --schedule cosine --metric el2n $PACED --f-min 0.2 --f-max 0.6 --seed $1
}

part_b() {      # seed
    method --schedule prune  --metric el2n $PACED --temperature 1 --seed $1
    method --schedule cosine --metric el2n $PACED --temperature 1 --seed $1
    method --schedule cosine --metric el2n $PACED --intro-rank descending --seed $1
    method --schedule cosine --metric loss $PACED --intro-rank descending --seed $1
}

part_a 1;   part_b 1;   python scratch_bidirectional.py --summarize
part_a 2;   part_b 2;   python scratch_bidirectional.py --summarize
