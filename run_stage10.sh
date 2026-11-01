#!/bin/bash
#==================================================================================================#
# STAGE 10: CONFIRMATION - THE METHOD WITH AN ESTIMATED NOISE RATE                                 #
#==================================================================================================#
# The method (--budget rule): train on full data to epoch 8, read p_hat = (m - e) / (1 - e) from    #
# the training negative-margin fraction m and the error e on a 2,000-sample trusted split carved   #
# from TRAINING data (never the test set), then fix budget = clip(p_hat + 0.3, 0.3, 0.8) and use    #
# EL2N-EMA hard exclusion at T = 1 if p_hat >= 0.1, else random exclusion. Pruning starts at 11.   #
#                                                                                                  #
# Every arm uses the same --trusted-size, so all arms train on exactly the same 48,000 samples.   #
# Baselines per condition: full data CE, full data GCE, random exclusion at the oracle budget.    #
#                                                                                                  #
# Part A: seed 1, both datasets (38 runs, ~16 h).  -> send confirm_summary.txt                     #
# Part B: sensitivity of the rule to a wrong p, CIFAR-100, seed 1 (10 runs, ~3.5 h).              #
# Part C: seeds 2-3 of Part A (76 runs) + sensitivity seed 2 (10 runs).                            #
# Each part ends with a summary; safe to stop between runs (Ctrl-C) and re-run to resume.         #
#==================================================================================================#

set -e

TRUST="--trusted-size 2000"
PRUNE="--epochs 100 --onset 10 --ramp 10 --stratify class --mode temporary $TRUST"
METHOD="--budget rule --score el2n --refresh ema --drop high"

condition() {   # dataset seed oracle_rho noise_args...
    local ds=$1 seed=$2 rho=$3; shift 3
    local B="--network resnet-18 --dataset $ds --seed $seed"
    python scratch_prune_ablation.py $B --epochs 100 --prune-fraction 0 $TRUST "$@"
    if [ "$*" != "" ]; then python scratch_prune_ablation.py $B --epochs 100 --prune-fraction 0 --objective gce $TRUST "$@"; fi
    python scratch_prune_ablation.py $B $PRUNE --prune-fraction $rho --score random "$@"
    python scratch_prune_ablation.py $B $PRUNE $METHOD "$@"
}

replicate() {   # seed
    for ds in cifar-100 cifar-10; do
        condition $ds $1 0.3
        condition $ds $1 0.5 --label-noise 0.2
        condition $ds $1 0.7 --label-noise 0.4
        condition $ds $1 0.5 --label-noise 0.2 --noise-type pairflip
        condition $ds $1 0.7 --label-noise 0.4 --noise-type pairflip
    done
}

sensitivity() { # seed: the rule fed a fixed (wrong or right) noise rate instead of the estimate
    local B="--network resnet-18 --dataset cifar-100 --seed $1"
    for p in 0.0 0.1 0.2 0.3 0.4; do
        python scratch_prune_ablation.py $B $PRUNE $METHOD --rule-noise $p --label-noise 0.2
    done
    for p in 0.0 0.2 0.3 0.4 0.5; do
        python scratch_prune_ablation.py $B $PRUNE $METHOD --rule-noise $p --label-noise 0.4 --noise-type pairflip
    done
}

summary() {
    python scratch_prune_ablation.py --summarize --network resnet-18 --dataset cifar-100
    python scratch_prune_ablation.py --summarize --network resnet-18 --dataset cifar-10
    python scratch_confirm_summary.py
}

# Part A
replicate 1;    summary

# Part B
sensitivity 1;  summary

# Part C
replicate 2;    replicate 3;    sensitivity 2;  summary
