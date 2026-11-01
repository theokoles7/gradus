#!/bin/bash
#==================================================================================================#
# BI-DIRECTIONAL PACING, STAGE 3: UNBIASED (RESCALED) SAMPLING  (clean CIFAR-100, ResNet-18)        #
#==================================================================================================#
# Stages 1-2: dropping samples without compensation loses to the compute-matched short schedule    #
# at every budget. InfoBatch (ICLR 2024) keeps the gradient unbiased by up-weighting the kept       #
# low-loss samples. This stage tests that mechanism, adds InfoBatch as the baseline, and puts the  #
# accuracy-paced bi-directional curves on top of the same unbiased sampler.                        #
#                                                                                                  #
#   1. InfoBatch baseline (loss, most recent value, ratio 0.5, full data for the last 12.5%)       #
#   2. cosine 1 <-> 0.5, loss, accuracy-paced, sampled + rescaled            <- the method         #
#   3. same, WITHOUT rescaling                                               (isolates rescaling)  #
#   4. same, fixed epoch clock                                               (isolates the signal) #
#   5. prune 1 -> 0.5, loss, accuracy-paced, sampled + rescaled              (direction ablation)  #
#   6. cosine 1 <-> 0.5, EL2N, accuracy-paced, sampled + rescaled            (metric ablation)     #
# Each: + compute-matched short schedule; rescaled arms + replay (unbiased uniform subsampling).    #
# ~6 h per seed. Safe to Ctrl-C and re-run to resume; earlier runs are reused.                     #
#==================================================================================================#

set -e

PACED="--clock signal --signal train_acc"
UNBIASED="--membership sampled --rescale"

method() {      # args...: run a method, then its controls
    python scratch_bidirectional.py "$@"
    python scratch_bidirectional.py "$@" --control short
    case "$*" in *"--rescale"*) python scratch_bidirectional.py "$@" --control replay ;; esac
}

stage() {       # seed
    method --schedule infobatch --metric loss --alpha 1 --seed $1
    method --schedule cosine --metric loss $PACED $UNBIASED --seed $1
    method --schedule cosine --metric loss $PACED --membership sampled --seed $1
    method --schedule cosine --metric loss --clock epoch $UNBIASED --seed $1
    method --schedule prune  --metric loss $PACED $UNBIASED --seed $1
    method --schedule cosine --metric el2n $PACED $UNBIASED --seed $1
}

stage 1;    python scratch_bidirectional.py --summarize
stage 2;    python scratch_bidirectional.py --summarize
