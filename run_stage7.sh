#!/bin/bash
#==================================================================================================#
# PRUNING ABLATION - STAGE 7: COMBINING WITH GCE, THIRD SEEDS, & NOISE-ESTIMATOR CALIBRATION       #
#==================================================================================================#
# (A) GCE + hard tilt vs. GCE + random: do selection & a robust loss stack?                        #
# (B) Third seed for the headline results (and their random / full-data references).               #
# Every new run also logs two online noise-rate estimates per epoch (negative-margin fraction &    #
# 2-component GMM), to calibrate the adaptive-budget rule next stage. Earlier runs are reused.     #
# 23 new runs (~9 h); safe to stop and re-run to resume.                                           #
#==================================================================================================#

set -e

BASE="--network resnet-18 --dataset cifar-100 --epochs 100"
PRUNE="--onset 10 --ramp 10 --stratify class --mode temporary"
HARD="--score el2n --refresh ema --drop high --temperature 1"
PF="--label-noise 0.2 --noise-type pairflip"

# (A) GCE + selection, seeds 1-2.
for seed in 1 2; do
    S="$BASE --seed $seed --objective gce"
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 --score random --label-noise 0.2
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD          --label-noise 0.2
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 --score random $PF
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD          $PF
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 --score random --label-noise 0.4
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 $HARD          --label-noise 0.4
done

# (B) Third seed for the headline comparisons (CE unless noted).
S="$BASE --seed 3"
python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD                  # clean, T=1 (estimator on clean data)
python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD --label-noise 0.2
python scratch_prune_ablation.py $S --prune-fraction 0 $PF
python scratch_prune_ablation.py $S --prune-fraction 0 $PF --objective gce
python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 --score random $PF
python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD $PF
python scratch_prune_ablation.py $S --prune-fraction 0 --label-noise 0.4
python scratch_prune_ablation.py $S --prune-fraction 0 --label-noise 0.4 --objective gce
python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 --score random --label-noise 0.4
python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 $HARD --label-noise 0.4
python scratch_prune_ablation.py $S --prune-fraction 0 --label-noise 0.2 --objective gce

python scratch_prune_ablation.py --summarize --network resnet-18 --dataset cifar-100