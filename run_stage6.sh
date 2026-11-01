#!/bin/bash
#==================================================================================================#
# PRUNING ABLATION - STAGE 6: STRUCTURED NOISE, A ROBUST-LOSS BASELINE, & THE (T, BUDGET) OPTIMUM  #
#==================================================================================================#
# (A) 40% symmetric noise: does a lower temperature / larger budget recover the under-filtering?   #
# (B) 20% symmetric noise: edges of the optimum (T=0.5; T=1 at 70%).                               #
# (C) 20% pairflip noise (structured, class-dependent): does the effect survive?                   #
# (D) Generalized Cross-Entropy on full data: the standard cheap robust-loss baseline.             #
# Earlier runs are reused (skipped). 28 new runs (~11-12 h); safe to stop and re-run to resume.    #
#==================================================================================================#

set -e

BASE="--network resnet-18 --dataset cifar-100 --epochs 100"
PRUNE="--onset 10 --ramp 10 --stratify class --mode temporary"
HARD="--score el2n --refresh ema --drop high"

for seed in 1 2; do

    S="$BASE --seed $seed"

    # (A) 40% symmetric noise.
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 --score random     --label-noise 0.4
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD --temperature 1 --label-noise 0.4
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 $HARD --temperature 1 --label-noise 0.4
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 $HARD --temperature 2 --label-noise 0.4

    # (B) 20% symmetric noise: optimum edges.
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD --temperature 0.5 --label-noise 0.2
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.7 $HARD --temperature 1   --label-noise 0.2

    # (C) 20% pairflip noise: full data, random, and the temperature curve.
    PF="--label-noise 0.2 --noise-type pairflip"
    python scratch_prune_ablation.py $S --prune-fraction 0 $PF
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 --score random $PF
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD --temperature 0 $PF
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD --temperature 1 $PF
    python scratch_prune_ablation.py $S $PRUNE --prune-fraction 0.5 $HARD --temperature 2 $PF

    # (D) GCE on full data (robust-loss baseline).
    python scratch_prune_ablation.py $S --prune-fraction 0 --objective gce --label-noise 0.2
    python scratch_prune_ablation.py $S --prune-fraction 0 --objective gce --label-noise 0.4
    python scratch_prune_ablation.py $S --prune-fraction 0 --objective gce $PF

done

python scratch_prune_ablation.py --summarize --network resnet-18 --dataset cifar-100