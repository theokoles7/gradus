#!/bin/bash
#==================================================================================================#
# BI-DIRECTIONAL PACING, STAGE 4: BUDGET-AWARE TRAINING  (clean CIFAR-100, ResNet-18)              #
#==================================================================================================#
# Stages 1-3 ran a 100-epoch LR schedule while processing 60-80% of the data; every arm (InfoBatch  #
# included) trailed the compute-matched short schedule. Here every arm spends the SAME sample       #
# budget and anneals its learning rate (cosine over samples processed) fully within it, so the     #
# budgeted short schedule (full data, B x 100 epochs) differs from each arm ONLY in which samples  #
# are trained on and when. The comparison column is vs_budget_full.                                #
#                                                                                                  #
# Per budget (75% and 50%):                                                                        #
#   reference   full data under the budget                                                         #
#   1. cosine 1 <-> 0.5, loss, sampled + rescaled, fixed clock          + replay                   #
#   2. same, accuracy-paced clock                                       + replay                   #
#   3. prune 1 -> 0.5, loss, sampled + rescaled                         + replay                   #
#   4. prune 1 -> 0.5, EL2N, pools (no rescaling; best unweighted arm)  + replay                   #
#   5. InfoBatch under the budget                                                                  #
# ~6 h per seed. Safe to Ctrl-C and re-run to resume.                                              #
#==================================================================================================#

set -e

PACED="--clock signal --signal train_acc"
UNBIASED="--membership sampled --rescale"

method() {      # args...: run a method, then its random replay
    python scratch_bidirectional.py "$@"
    python scratch_bidirectional.py "$@" --control replay
}

stage() {       # seed budget
    local B="--seed $1 --budget $2"
    python scratch_bidirectional.py --schedule full $B
    method --schedule cosine --metric loss --clock epoch $UNBIASED $B
    method --schedule cosine --metric loss $PACED $UNBIASED $B
    method --schedule prune  --metric loss --clock epoch $UNBIASED $B
    method --schedule prune  --metric el2n $PACED $B
    python scratch_bidirectional.py --schedule infobatch --metric loss --alpha 1 $B
}

stage 1 0.75;   stage 1 0.5;    python scratch_bidirectional.py --summarize
stage 2 0.75;   stage 2 0.5;    python scratch_bidirectional.py --summarize
