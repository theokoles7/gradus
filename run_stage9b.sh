#!/bin/bash
#==================================================================================================#
# STAGE 9b: THE METHOD WITH AN ORACLE NOISE RATE (upper bound & rule validation)                   #
#==================================================================================================#
# Rule (fitted on CIFAR-100 stages 5-6):  budget = clip(p + 0.3, 0.3, 0.8);  T = 1 if p >= 0.1 else 2  #
#   p=0.0 -> 0.3 / T=2   p=0.1 -> 0.4 / T=1   p=0.2 -> 0.5 / T=1   p=0.3 -> 0.6 / T=1   p=0.4 -> 0.7 / T=1 #
# Each condition: the rule, random exclusion at the same budget, full data (CE), full data (GCE).  #
# Part A fills the untested CIFAR-100 noise levels; Part B is an out-of-sample test on CIFAR-10.   #
# Earlier runs are reused (skipped). Part A: 24 runs (~10 h). Part B: 22 runs (~8 h).             #
# Safe to stop after Part A and send its summary; re-run to resume.                                #
#==================================================================================================#

set -e

PRUNE="--epochs 100 --onset 10 --ramp 10 --stratify class --mode temporary"
HARD="--score el2n --refresh ema --drop high"

condition() {   # dataset network seed rho T noise_args...
    local ds=$1 net=$2 seed=$3 rho=$4 T=$5; shift 5
    local B="--network $net --dataset $ds --seed $seed"
    python scratch_prune_ablation.py $B --epochs 100 --prune-fraction 0 "$@"
    if [ "$*" != "" ]; then python scratch_prune_ablation.py $B --epochs 100 --prune-fraction 0 --objective gce "$@"; fi
    python scratch_prune_ablation.py $B $PRUNE --prune-fraction $rho --score random "$@"
    python scratch_prune_ablation.py $B $PRUNE --prune-fraction $rho $HARD --temperature $T "$@"
}

# Part A: CIFAR-100, untested noise levels.
for seed in 1 2; do
    condition cifar-100 resnet-18 $seed 0.4 1 --label-noise 0.1
    condition cifar-100 resnet-18 $seed 0.6 1 --label-noise 0.3
    condition cifar-100 resnet-18 $seed 0.7 1 --label-noise 0.4 --noise-type pairflip
done
python scratch_prune_ablation.py --summarize --network resnet-18 --dataset cifar-100

# Part B: CIFAR-10, out-of-sample check of the same rule.
for seed in 1 2; do
    condition cifar-10 resnet-18 $seed 0.3 2
    condition cifar-10 resnet-18 $seed 0.5 1 --label-noise 0.2
    condition cifar-10 resnet-18 $seed 0.7 1 --label-noise 0.4
done
python scratch_prune_ablation.py --summarize --network resnet-18 --dataset cifar-10
