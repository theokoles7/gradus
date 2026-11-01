#!/usr/bin/env python3
"""
scratch_bpas.py

Batch Pruning by Activation Stability (B-PAS), reimplemented faithfully, plus
matched-count control arms that swap the SELECTION CRITERION while holding the
pruning schedule fixed.

Reference: Alam, Islam & Islam, "Batch Pruning by Activation Stability",
ICLR 2026. CIFAR settings from that paper: ResNet-18, 200 epochs, batch 128,
lr 0.10, weight decay 5e-4, CosineAnnealing, delta in [1e-6, 5e-5]. Reported
CIFAR-100 ResNet-18 full-dataset baseline 78.20 +- 0.3, B-PAS 78.20 +- 0.1 at
24% DSI, random pruning at matched DSI 75.36 +- 0.4.

Method as specified in that paper
---------------------------------
FIXED GLOBAL BATCHES. The corpus is partitioned into n fixed batches before
training. A sample assigned to batch B_m stays in B_m for the whole run unless
B_m is pruned. The trailing residual batch is dropped. Each epoch shuffles the
order of samples within each batch and the order in which batches are loaded,
but never moves a sample between batches.

    NOTE: intra-batch shuffling does not affect the gradient, since the
    minibatch gradient is invariant to permutation within the batch. This
    project measured that directly: 242 batch-wise curriculum runs came in at
    -0.02pp against the unshuffled baseline. So the stochasticity in this
    protocol comes from batch ORDER alone, while the full-dataset baseline
    reshuffles composition every epoch. That is a difference between arms, not
    a neutral implementation detail, which is why the fixed_full arm below
    exists.

ACTIVATION TRACKING. For batch B_m at epoch i, the output of each residual
stage is flattened and its standard deviation taken,

    sigma^(i)_{j,m} = sqrt( (1/N) sum_k (t_k - mu)^2 ),   N = b*C*H*W

then averaged over the l stages to one scalar per batch per epoch,

    Xbar_i(B_m) = (1/l) sum_j sigma^(i)_{j,m}.

Stage outputs are post-ReLU in a ResNet basic block, which is what the paper
specifies. Statistics are collected during the ordinary training forward pass,
so tracking is free.

PRUNING CRITERION. From the second epoch, a batch is pruned when

    |Xbar_i(B_m) - Xbar_{i-1}(B_m)| <= delta(i),
    delta(i) = delta_s * exp(alpha * i),  alpha = (1/I) ln(delta_e / delta_s)

with I the scheduled epoch count. Pruning is permanent and the active batch
count decreases monotonically.

DSI = 1 - sum_{i=1..es} n_i / (e0 * n0).

Arms
----
  baseline        full data, standard per-epoch reshuffling. The paper's
                  baseline.
  reshuffled_390  full data, composition RESHUFFLED every epoch, but the
                  trailing residual dropped so there are 390 equal batches
                  rather than 391. Separates the two ways fixed_full differs
                  from baseline: fixed composition, and the dropped residual.
                  Without it, "fixed partitioning costs 1.39pp" conflates the
                  two.
  fixed_full      full data, FIXED partition, no pruning. Isolates the cost of
                  fixed batch composition from the cost of pruning. The paper
                  does not run this; it is the control that makes "no accuracy
                  loss" interpretable.
  bpas            activation-stability pruning as above.
  random_matched  prunes the SAME NUMBER of batches per epoch as a reference
                  B-PAS run, chosen uniformly at random. This is the paper's
                  own random control ("we track the number of batches flagged
                  by activation stability but prune the same number of batches
                  chosen at random").
  complexity      same matched counts, batches chosen by mean complexity
                  metric instead. The substitution under test.

  sample_random   SAMPLE-level pruning with composition RESHUFFLED every epoch,
                  class-balanced, at a matched DSI. Batch-level pruning needs
                  fixed batches because the activation statistic is per-batch
                  across epochs; a sample-level criterion does not, so it can
                  keep the stochasticity that fixed composition destroys.
  sample_accuracy same, but the retained set is chosen by per-sample accuracy
                  EMA (lowest accuracy kept, most-mastered pruned) rather than
                  at random.
  sample_complexity same, selected by a precomputed curriculum complexity
                  metric. This is the only cell where a complexity criterion
                  has room to show anything: every earlier complexity run sat
                  inside the fixed-composition protocol, which costs 2.4 to
                  3.6pp at matched DSI and swamps any criterion signal, and
                  was run at DSI points where random selection costs almost
                  nothing. Sample-level, reshuffled, high DSI is where a
                  selection criterion can matter if it matters at all.

  The motivating measurement: fixed composition alone costs 1.54pp (sd 0.12,
  0/3) with no pruning at all, while dropping the residual costs nothing
  (+0.15pp). So at matched DSI the sample-level arms should beat the
  batch-level ones by roughly that protocol gap. Whether the CRITERION adds
  anything on top is the separate question, and this project's prior
  measurements say probably not.

  The paper's own ablation reports sample-level pruning at 70.87 against
  batch-level at 78.43 on ImageNet and attributes the gap to class imbalance.
  These arms mask per class, which is the control for that.

random_matched and complexity require --schedule-from pointing at a finished
bpas run, so the batch-count trajectory is identical and only the criterion
differs.

Expected weakness of the complexity arm, stated in advance
----------------------------------------------------------
Activation stability is a genuine per-batch property: the paper's Figure 5
shows batch trajectories separating. A batch's MEAN complexity over 128
randomly assigned samples concentrates within about sigma/sqrt(128), roughly
9% of the sample-level spread, so there may be very little to select on. This
project hit the same wall measuring batch-mean ordering signals. Composing
batches BY the metric instead (--batch-composition by-metric) gives a strong
between-batch signal, but makes batches difficulty-homogeneous, which cost
0.82pp at full data in this project's own decomposition. Both are available;
neither is free.

Usage
-----
    python scratch_bpas.py --self-test
    python scratch_bpas.py --seed 1 --arm bpas \\
        --output-path results/bpas/bpas_seed1.json
    python scratch_bpas.py --seed 1 --arm complexity \\
        --schedule-from results/bpas/bpas_seed1.json \\
        --static-scores .cache/scores/metric-scores_seed-1.parquet \\
        --static-metric saturation-time --prune-side low \\
        --output-path results/bpas/complexity_sat_seed1.json
    python scratch_bpas.py --analyze "results/bpas/*.json"
"""

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

import numpy as np

torch = None
nn = None
CosineAnnealingLR = None
ResNet18 = None
CIFAR_100 = None


def _lazy_imports():
    global torch, nn, CosineAnnealingLR, ResNet18, CIFAR_100

    import torch as _t
    import torch.nn as _nn
    from torch.optim.lr_scheduler import CosineAnnealingLR as _Cos
    # RECONCILE: confirm these against the installed package layout.
    from gradus.networks.resnet.resnet_18 import ResNet18 as _R
    from gradus.datasets.cifar_100 import CIFAR_100 as _D

    torch, nn, CosineAnnealingLR = (_t, _nn, _Cos)
    ResNet18, CIFAR_100 = _R, _D


# Paper's CIFAR-100 settings
BASE_LR = 0.10
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
EPOCHS = 200
NUM_CLASSES = 100
DELTA_S = 1e-6
DELTA_E = 5e-5

# Transforms, normalisation, data root, worker count and the test loader all
# come from gradus.datasets.cifar_100.CIFAR_100. Nothing is redefined here.

ARMS = ("baseline", "reshuffled_390", "fixed_full", "bpas",
        "random_matched", "complexity",
        "sample_random", "sample_accuracy", "sample_complexity")
MATCHED = ("random_matched", "complexity",
           "sample_random", "sample_accuracy", "sample_complexity")
SAMPLE_ARMS = ("sample_random", "sample_accuracy", "sample_complexity")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(epochs, seed):
    """Gradus CIFAR_100, not a local reimplementation.

    metric=None routes Dataset.train_loader to a plain shuffled DataLoader with
    no curriculum sampler, which is the established baseline path. shuffle=True
    is passed explicitly because the Gradus default is False. Transforms,
    normalisation constants, batch size, worker count and the test loader all
    come from the registered dataset, so nothing here can drift from the rest
    of the corpus.
    """
    return CIFAR_100(epochs=epochs, batch_size=BATCH_SIZE, shuffle=True,
                     metric=None, schedule=None, seed=seed)


def delta_schedule(epoch, delta_s, delta_e, total_epochs):
    """delta(i) = delta_s * exp(alpha * i), alpha = (1/I) ln(delta_e/delta_s).

    Conservative early, when representations are still changing, and
    progressively more permissive as the model converges.
    """
    alpha = math.log(delta_e / delta_s) / total_epochs
    return delta_s * math.exp(alpha * epoch)


def fixed_partition(n_total, bs, rng, scores=None, composition="random"):
    """Fixed global batches, formed once. The trailing residual is dropped so
    every batch has the same size, as the paper specifies.

    composition 'random'     randomly assigned, mixed batches (paper default)
    composition 'by-metric'  sorted by score, so batches are metric-homogeneous
                             and the between-batch signal is strong. Costs the
                             composition penalty measured in this project.
    """
    nb = n_total // bs
    if composition == "by-metric":
        if scores is None:
            raise SystemExit("by-metric composition needs --static-scores")
        order = np.argsort(scores, kind="stable")[:nb * bs]
    else:
        order = rng.permutation(n_total)[:nb * bs]
    return [np.sort(order[t * bs:(t + 1) * bs]) for t in range(nb)]


class StageTap:
    """Captures the four residual-stage outputs of the Gradus ResNet via its
    own return_activations path (already detached, post-ReLU).
    """

    @staticmethod
    def batch_statistic(activations):
        """Xbar = mean over stages of the std of the flattened stage output."""
        return float(np.mean([float(a.float().std()) for a in activations]))


def dsi(active_counts, n0, e0):
    """DSI = 1 - sum_i n_i / (e0 * n0)."""
    return 1.0 - sum(active_counts) / (e0 * n0)



def dsi_schedule(n0, epochs, dsi_target, keep_floor=0.10, tol=1e-6):
    """Per-epoch target active-batch counts that realise a given DSI.

    B-PAS's DSI is an OUTCOME of delta, not something you set, which makes
    matched comparisons awkward and makes a frontier impossible to draw. For
    the criterion comparison we instead fix DSI and vary only which batches
    are removed, so accuracy-vs-DSI curves for different criteria are directly
    comparable at every point.

    Shape: geometric decay of the active count from epoch 2, floored at
    keep_floor * n0, matching the paper's qualitative trajectory (conservative
    early, accelerating later, monotone). The decay rate p is found by
    bisection so the realised DSI hits the target.

        DSI = 1 - sum_i n_i / (epochs * n0)
    """
    floor_n = max(1, int(round(keep_floor * n0)))

    def build(p):
        cnt, cur = [n0], n0
        for _ in range(epochs - 1):
            cur = max(floor_n, int(round(cur * (1.0 - p))))
            cnt.append(cur)
        return cnt

    lo, hi = 0.0, 0.5
    if 1.0 - sum(build(hi)) / (epochs * n0) < dsi_target:
        raise SystemExit(
            f"DSI {dsi_target:.2f} unreachable with keep_floor="
            f"{keep_floor}; max is "
            f"{1.0 - sum(build(hi)) / (epochs * n0):.3f}. Lower the floor.")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - sum(build(mid)) / (epochs * n0) < dsi_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    cnt = build(hi)
    return cnt, 1.0 - sum(cnt) / (epochs * n0), hi



def sample_mask(scores, targets, keep_frac, mode, rng, n_classes):
    """Indices RETAINED, chosen per class so pruning cannot distort the class
    distribution. That is the stated failure mode of sample-level pruning in
    the B-PAS ablation (70.87 vs 78.43 on ImageNet), so it is controlled here
    rather than left to chance.

    mode 'random'   uniform within each class
    mode 'lowest'   keep the LOWEST-scoring samples, i.e. prune the top of the
                    (already sign-adjusted) score. For accuracy EMA that
                    prunes the most mastered; for a complexity metric with
                    --prune-side low it prunes the easiest.
    """
    keep = []
    for c in range(n_classes):
        idx = np.where(targets == c)[0]
        k = int(round(keep_frac * len(idx)))
        if k <= 0:
            continue
        if mode == "random":
            keep.append(rng.choice(idx, size=k, replace=False))
        else:
            keep.append(idx[np.argsort(scores[idx], kind="stable")[:k]])
    return np.sort(np.concatenate(keep))


def load_parquet_scores(path, metric, n_expected, prune_side):
    import pandas as pd

    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c not in ("index", "class")]
    if metric not in df.columns:
        raise SystemExit(f"'{metric}' not in {path}. Available: {', '.join(cols)}")
    df = df.sort_values("index")
    idx = df["index"].to_numpy()
    if idx[0] != 0 or idx[-1] != len(idx) - 1:
        raise SystemExit("`index` is not a contiguous 0..N-1 range")
    v = df[metric].to_numpy().astype(np.float32)
    if len(v) != n_expected:
        raise SystemExit(f"{metric}: {len(v)} rows != {n_expected}")
    ties = len(v) - len(np.unique(v))
    print(f"  static score: {metric} [min {v.min():.4g}, max {v.max():.4g}, "
          f"unique {len(np.unique(v))}, tied {ties}], pruning the "
          f"{prune_side} end first")
    return -v if prune_side == "low" else v


class IndexBatchLoader:
    def __init__(self, dataset, batches, device, rng):
        self.dataset, self.batches, self.device, self.rng = (
            dataset, batches, device, rng)

    def __iter__(self):
        for bid, idx in self.batches:
            # intra-batch shuffle, per the paper's protocol. Does not change
            # the batch gradient; included for fidelity.
            idx = idx[self.rng.permutation(len(idx))]
            xs = torch.stack([self.dataset[int(i)][0] for i in idx])
            ys = torch.tensor([self.dataset[int(i)][1] for i in idx])
            yield bid, xs.to(self.device, non_blocking=True), \
                ys.to(self.device, non_blocking=True), idx


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def run(args):
    _lazy_imports()
    if os.path.exists(args.output_path):
        raise SystemExit(f"{args.output_path} exists. Use a distinct path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(args.epochs, args.seed)
    ds_train = ds.train_data          # augmented training split
    test_loader = ds.test_loader      # Gradus test loader, unchanged
    n_total = len(ds_train)

    scores = None
    if args.static_scores:
        scores = load_parquet_scores(args.static_scores, args.static_metric,
                                     n_total, args.prune_side)

    prune_counts, targets = None, None
    if args.arm in MATCHED:
        if bool(args.schedule_from) == bool(args.dsi_target):
            raise SystemExit(f"{args.arm} needs exactly one of "
                             f"--schedule-from or --dsi-target")
        if args.schedule_from:
            ref = json.load(open(args.schedule_from))
            if ref["arm"] != "bpas":
                raise SystemExit("--schedule-from must point at a bpas run")
            prune_counts = [e["pruned_this_epoch"] for e in ref["epochs_log"]]
            print(f"  matched schedule from "
                  f"{os.path.basename(args.schedule_from)}: "
                  f"{sum(prune_counts)} batches over {len(prune_counts)} epochs")
        else:
            nb0 = len(ds_train) // BATCH_SIZE
            targets, realised, rate = dsi_schedule(
                nb0, args.epochs, args.dsi_target, args.keep_floor)
            print(f"  DSI schedule: target {args.dsi_target:.2f}, realised "
                  f"{realised:.4f}, decay {rate:.5f}/epoch, "
                  f"{nb0} -> {targets[-1]} batches")

    set_seed(args.seed)
    # same construction as gradus.commands.train: shape and class count come
    # from the dataset, optimizer is SGD(lr=0.1, wd=5e-4, momentum=0.9)
    model = ResNet18(input_shape=ds.input_shape,
                     num_classes=ds.num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    prng = np.random.default_rng(args.seed * 1000 + 31)
    brng = np.random.default_rng(args.seed * 1000 + 7)

    # reshuffled_390 repartitions every epoch like the baseline, but drops the
    # residual like the fixed arms, so it carries exactly one of the two
    # differences between baseline and fixed_full.
    reshuffle_390 = args.arm == "reshuffled_390"
    sample_level = args.arm in SAMPLE_ARMS
    fixed = args.arm not in ("baseline", "reshuffled_390") and not sample_level
    if fixed:
        parts = fixed_partition(n_total, BATCH_SIZE, prng, scores,
                                args.batch_composition)
        active = list(range(len(parts)))
        n0 = len(parts)
        batch_scores = (np.array([scores[p].mean() for p in parts])
                        if scores is not None else None)
    elif reshuffle_390 or sample_level:
        n0 = n_total // BATCH_SIZE
    else:
        n0 = len(ds.train_loader)

    prev_stat = {}
    log, active_counts = [], []
    best = 0.0

    if sample_level:
        labels = np.asarray(ds_train.targets)
        n_classes = ds.num_classes
        retained = np.arange(n_total)
        acc_ema = np.full(n_total, np.nan, dtype=np.float64)
        srng = np.random.default_rng(args.seed * 1000 + 77)

    for epoch in range(1, args.epochs + 1):
        if fixed:
            order = brng.permutation(len(active))
            batches = [(active[i], parts[active[i]]) for i in order]
        elif sample_level:
            # composition RESHUFFLED every epoch over the retained set, which
            # is what batch-level pruning cannot do
            nb = len(retained) // BATCH_SIZE
            perm = retained[brng.permutation(len(retained))][:nb * BATCH_SIZE]
            batches = [(-1, perm[t * BATCH_SIZE:(t + 1) * BATCH_SIZE])
                       for t in range(nb)]
        elif reshuffle_390:
            nb = n_total // BATCH_SIZE
            perm = brng.permutation(n_total)[:nb * BATCH_SIZE]
            batches = [(-1, perm[t * BATCH_SIZE:(t + 1) * BATCH_SIZE])
                       for t in range(nb)]
        else:
            # baseline: the Gradus train_loader itself, so this arm is the
            # established baseline path rather than a copy of it. drop_last is
            # False there, so it yields 391 batches including a short final
            # one, unlike the 390 equal batches the fixed-partition arms use.
            batches = None
        active_counts.append(len(batches) if batches is not None
                             else len(ds.train_loader))

        model.train()
        rl, seen = 0.0, 0
        stat = {}
        stream = (((-1, x.to(device), y.to(device), None)
                   for x, y in ds.train_loader) if batches is None
                  else IndexBatchLoader(ds_train, batches, device, brng))
        for bid, x, y, bid_idx in stream:
            opt.zero_grad(set_to_none=True)
            if args.arm == "bpas":
                out, acts = model(x, return_activations=True)
                stat[bid] = StageTap.batch_statistic(acts)
            else:
                out = model(x)
            if sample_level and args.arm == "sample_accuracy":
                with torch.no_grad():
                    corr = (out.argmax(1) == y).float().cpu().numpy()
                ids = np.asarray(bid_idx)
                prev = acc_ema[ids]
                fresh = np.isnan(prev)
                acc_ema[ids] = np.where(
                    fresh, corr,
                    args.acc_ema * np.nan_to_num(prev) + (1 - args.acc_ema) * corr)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            rl += loss.item() * y.size(0)
            seen += y.size(0)

        pruned = 0
        if sample_level and epoch >= 2:
            nxt = targets[epoch] if epoch < len(targets) else targets[-1]
            keep_frac = (nxt * BATCH_SIZE) / n_total
            before = len(retained)
            if args.arm == "sample_accuracy":
                sc = acc_ema
            elif args.arm == "sample_complexity":
                if scores is None:
                    raise SystemExit("sample_complexity needs --static-scores")
                sc = scores
            else:
                sc = np.zeros(n_total)
            # keep the lowest scores: for accuracy that prunes the mastered
            retained = sample_mask(np.nan_to_num(sc, nan=-1.0), labels,
                                   keep_frac, 
                                   "random" if args.arm == "sample_random"
                                   else "lowest", srng, n_classes)
            pruned = max(0, (before - len(retained)) // BATCH_SIZE)
        if fixed and args.arm != "fixed_full" and epoch >= 2:
            if args.arm == "bpas":
                d = delta_schedule(epoch, args.delta_s, args.delta_e,
                                   args.epochs)
                cand = [b for b in active
                        if b in prev_stat and b in stat
                        and abs(stat[b] - prev_stat[b]) <= d]
                # B-PAS as published has no floor on the active set, so a
                # widened delta_e eventually clears it and the run hits the
                # len(active) == 0 break below before the cosine schedule has
                # finished annealing. Measured: at delta_e=4e-4 the three seeds
                # stopped at epochs 180, 176 and 175 of 200, and the anneal is
                # worth several points (one trajectory went 66.01 at epoch 120
                # to 75.06 at 140). Accuracy in those cells therefore conflates
                # data removal with truncated annealing.
                #
                # --bpas-keep-floor caps removal so the run completes the
                # schedule, dropping the MOST stable candidates first, which is
                # the criterion's own ordering: the retained set is still the
                # one activation stability ranks least stable. The default is
                # 0.0, which reproduces the published behaviour exactly, so the
                # faithful arm is unchanged and the floored series is opt-in.
                floor_n = max(1, int(round(args.bpas_keep_floor * n0))) \
                    if args.bpas_keep_floor > 0 else 0
                room = max(0, len(active) - floor_n)
                if floor_n and len(cand) > room:
                    cand.sort(key=lambda b: abs(stat[b] - prev_stat[b]))
                    cand = cand[:room]
                drop = cand
            else:
                if targets is not None:
                    nxt = (targets[epoch] if epoch < len(targets)
                           else targets[-1])
                    k = max(0, len(active) - nxt)
                else:
                    k = (prune_counts[epoch - 1]
                         if epoch - 1 < len(prune_counts) else 0)
                k = min(k, max(len(active) - 1, 0))
                if k <= 0:
                    drop = []
                elif args.arm == "random_matched":
                    drop = list(prng.choice(active, size=k, replace=False))
                else:   # complexity: prune the top of the (possibly negated) score
                    cand = sorted(active, key=lambda b: -batch_scores[b])
                    drop = cand[:k]
            if drop:
                active = [b for b in active if b not in set(drop)]
                pruned = len(drop)
        prev_stat = stat

        acc = evaluate(model, test_loader, device)
        best = max(best, acc)
        log.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                    "train_loss": rl / seen, "test_acc": acc,
                    "active_batches": (len(batches) if batches is not None
                                       else len(ds.train_loader)),
                    "pruned_this_epoch": pruned,
                    "delta": (delta_schedule(epoch, args.delta_s, args.delta_e,
                                             args.epochs)
                              if args.arm == "bpas" else None)})
        if epoch % 20 == 0 or epoch == args.epochs:
            nb_now = len(batches) if batches is not None else len(ds.train_loader)
            print(f"ep {epoch:3d} loss={rl/seen:.4f} test={acc:.2f} "
                  f"best={best:.2f} active={nb_now}/{n0}")
        sched.step()
        if fixed and len(active) == 0:
            print(f"  all batches pruned at epoch {epoch}; stopping")
            break

    out = {"seed": args.seed, "arm": args.arm,
           "epochs_scheduled": args.epochs, "epochs_run": len(log),
           "batch_size": BATCH_SIZE, "n0_batches": n0,
           "fixed_partition": fixed,
           "dataset": ds.dict,
           "input_shape": list(ds.input_shape),
           "batch_composition": args.batch_composition if fixed else "n/a",
           "delta_s": args.delta_s, "delta_e": args.delta_e,
           "static_metric": args.static_metric,
           "sample_level": sample_level,
           "prune_side": args.prune_side,
           "schedule_from": args.schedule_from,
           "dsi_target": args.dsi_target,
           "keep_floor": args.keep_floor,
           "bpas_keep_floor": args.bpas_keep_floor,
           "dsi": dsi(active_counts, n0, args.epochs),
           "total_batches_pruned": n0 - (len(active) if fixed else n0),
           "best_test_acc": best, "final_test_acc": log[-1]["test_acc"],
           "epochs_log": log}
    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{args.arm} seed {args.seed}: best={best:.2f} "
          f"DSI={100*out['dsi']:.2f}%\n-> {args.output_path}")


def self_test():
    print("delta schedule, delta_s=1e-6 delta_e=5e-5 over 200 epochs:")
    for e in (1, 25, 50, 100, 150, 200):
        print(f"  epoch {e:>3}: delta = {delta_schedule(e, 1e-6, 5e-5, 200):.3e}")
    assert abs(delta_schedule(200, 1e-6, 5e-5, 200) - 5e-5) < 1e-12
    print("  reaches delta_e exactly at the final epoch: True")

    N, bs = 50000, 128
    rng = np.random.default_rng(0)
    sc = rng.normal(size=N).astype(np.float32)
    for comp in ("random", "by-metric"):
        parts = fixed_partition(N, bs, np.random.default_rng(1), sc, comp)
        bm = np.array([sc[p].mean() for p in parts])
        print(f"\ncomposition={comp}: {len(parts)} batches, "
              f"{len(parts)*bs} samples ({N - len(parts)*bs} residual dropped)")
        print(f"  batch-mean score sd {bm.std():.4f} vs sample sd {sc.std():.4f} "
              f"(ratio {bm.std()/sc.std():.3f})")
        print(f"  sigma/sqrt(bs) = {sc.std()/math.sqrt(bs):.4f}")
        flat = np.sort(np.concatenate(parts))
        print(f"  every sample in exactly one batch: "
              f"{len(flat) == len(np.unique(flat))}")

    print("\n  -> with random composition the complexity arm has ~9% of the "
          "sample-level spread to select on; with by-metric it has nearly all "
          "of it, at the cost of homogeneous batches")

    print("\nDSI check (paper's worked example: 5 epochs, 200 batches, stops "
          "at epoch 3 after 200, 190, 180):")
    v = dsi([200, 190, 180], 200, 5)
    print(f"  DSI = {v:.4f}  (paper states 0.43): {abs(v - 0.43) < 0.005}")

    print("\nbpas keep-floor check (the live block's logic, 100 active "
          "batches of n0=390, every batch below threshold):")
    n0_t, active_t = 390, list(range(100))
    deltas = {b: 1e-9 * (b + 1) for b in active_t}   # b=0 most stable
    for kf in (0.0, 0.10, 0.25):
        cand = list(active_t)
        floor_n = max(1, int(round(kf * n0_t))) if kf > 0 else 0
        room = max(0, len(active_t) - floor_n)
        if floor_n and len(cand) > room:
            cand.sort(key=lambda b: deltas[b])
            cand = cand[:room]
        kept = sorted(set(active_t) - set(cand))
        print(f"  keep_floor={kf:<5} floor_n={floor_n:<3} "
              f"drop {len(cand):>3}, keep {len(kept):>3}")
        if kf == 0.0:
            assert len(kept) == 0, "floor 0 must reproduce unfloored collapse"
        else:
            assert len(kept) == min(len(active_t), floor_n)
            # the survivors must be the LEAST stable, i.e. largest |delta|
            assert min(deltas[b] for b in kept) > max(deltas[b] for b in cand)
    print("  floor 0 collapses the active set; a nonzero floor retains exactly "
          "floor_n batches and they are the least stable ones: True")

    print("\nself-test passed")


def cell_name(L):
    """Group key for a finished run.

    A bpas run at the published delta range keeps the bare name "bpas", so the
    primary-table contrasts below and any existing analysis path still find it.
    A widened delta_e or a nonzero keep floor is a different operating point on
    the same method, so it gets its own key. Keying on delta_e rather than on
    realised DSI matters: realised DSI varies by seed (16.01 to 18.29% at the
    published range), so bucketing the sweep on DSI puts every seed in a cell
    of its own and nothing pools.
    """
    name = L["arm"]
    if name in ("complexity", "sample_complexity") and L.get("static_metric"):
        return f"{name}:{L['static_metric']}"
    if name == "bpas":
        tag = ""
        if L.get("delta_e", DELTA_E) != DELTA_E:
            tag += f"@de{L['delta_e']:g}"
        if L.get("bpas_keep_floor", 0.0):
            tag += f"+floor{L['bpas_keep_floor']:g}"
        return name + tag
    # A matched arm run at several --dsi-target values is several operating
    # points, not one arm. Without the target in the key, by[name][seed]
    # overwrites and the summary table reports whichever file happened to sort
    # last under an unqualified label, at a DSI that belongs to only one cell.
    if L.get("dsi_target"):
        return f"{name}@d{L['dsi_target']:g}"
    return name


def analyze(paths):
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit("no files matched")
    by, meta = defaultdict(dict), defaultdict(dict)
    short = defaultdict(dict)
    for f in files:
        L = json.load(open(f))
        name = cell_name(L)
        by[name][L["seed"]] = L["best_test_acc"]
        meta[name][L["seed"]] = L["dsi"]
        # a run that ended before its scheduled epoch count forfeits part of
        # the cosine anneal, so accuracy in that cell is not attributable to
        # data removal alone
        if L.get("epochs_run", L["epochs_scheduled"]) < L["epochs_scheduled"]:
            short[name][L["seed"]] = L["epochs_run"]
    arms = sorted(by, key=lambda a: (a not in ARMS, a))
    seeds = sorted(set.intersection(*[set(by[a]) for a in arms]))
    print(f"{len(seeds)} shared seed(s): {seeds}\n")
    print(f"{'arm':<26}{'best acc':>12}{'DSI %':>10}")
    col = {}
    for a in arms:
        v = np.array([by[a][s] for s in seeds])
        d = np.array([meta[a][s] for s in seeds]) * 100
        col[a] = v
        sd = f" (sd {v.std(ddof=1):.2f})" if len(seeds) > 1 else ""
        flag = ""
        if a in short:
            eps = sorted(short[a].values())
            flag = (f"   TRUNCATED: {len(short[a])} seed(s) ended at "
                    f"epoch {eps[0]}-{eps[-1]}")
        print(f"{a:<26}{v.mean():>8.2f}{sd:<9}{d.mean():>8.2f}{flag}")
    if short:
        print("\n  a TRUNCATED cell ended before its scheduled epoch count, so "
              "it forfeited part of the cosine anneal and its accuracy mixes "
              "data removal with truncated annealing. For the bpas arm, rerun "
              "with --bpas-keep-floor to hold the run to full length.")

    n = len(seeds)

    def rep(label, d):
        m = d.mean()
        if n > 1:
            se = d.std(ddof=1) / math.sqrt(n)
            print(f"  {label:<40} {m:+6.2f}pp  sd={d.std(ddof=1):.2f}  "
                  f"t={m/se:+.2f} (df={n-1})  pos={int((d>0).sum())}/{n}")
        else:
            print(f"  {label:<40} {m:+6.2f}pp  (single seed)")

    if "baseline" in col:
        print("\nvs the paper's baseline (full data, reshuffled each epoch):")
        for a in arms:
            if a != "baseline":
                rep(f"{a} - baseline", col[a] - col["baseline"])
    if "fixed_full" in col and "baseline" in col:
        print("\nprotocol cost, decomposed (no pruning in any of these arms):")
        rep("fixed_full - baseline        both differences",
            col["fixed_full"] - col["baseline"])
        if "reshuffled_390" in col:
            rep("reshuffled_390 - baseline   dropped residual only",
                col["reshuffled_390"] - col["baseline"])
            rep("fixed_full - reshuffled_390 fixed composition only",
                col["fixed_full"] - col["reshuffled_390"])
        else:
            print("  no reshuffled_390 arm: this conflates fixed composition "
                  "with the 80 samples dropped by the residual")
        print("  a nonzero value here means the paper's pruned arms and its "
              "baseline differ in more than pruning")
    if "fixed_full" in col and "bpas" in col:
        print("\nB-PAS against the arm that shares its protocol:")
        rep("bpas - fixed_full", col["bpas"] - col["fixed_full"])
    if "bpas" in col:
        print("\nselection criterion, at matched batch-count trajectories:")
        for a in arms:
            if a.startswith("complexity") or a.startswith("random_matched"):
                rep(f"{a} - bpas", col[a] - col["bpas"])

    print("\nfrontier (accuracy against realised DSI, by arm):")
    front = defaultdict(lambda: defaultdict(list))
    for f in files:
        L = json.load(open(f))
        nm = cell_name(L)
        # Bucket on the DESIGNED axis, not the realised one. --dsi-target is
        # identical across seeds, so it identifies a cell exactly; realised DSI
        # is not, so bucketing on it splits a 3-seed cell into three 1-seed
        # cells. bpas has no target and cell_name has already separated its
        # operating points by delta_e, so all of its seeds pool into one cell.
        tgt = L.get("dsi_target")
        front[nm][round(100 * tgt, 1) if tgt else None].append(
            (100 * L["dsi"], L["best_test_acc"]))
    for nm in sorted(front):
        cells = []
        for _, pts in sorted(front[nm].items(),
                             key=lambda kv: np.mean([d for d, _ in kv[1]])):
            d = np.mean([x for x, _ in pts])
            a = np.mean([y for _, y in pts])
            cells.append(f"{d:.1f}%:{a:.2f}(n={len(pts)})")
        print(f"  {nm:<26} {'  '.join(cells)}")

    print("\nPaper reference (CIFAR-100, ResNet-18): baseline 78.20+-0.3, "
          "B-PAS 78.20+-0.1 at 24% DSI, random at matched DSI 75.36+-0.4.")
    print("Computed ad hoc by this script; rerun in your own analysis code "
          "before citing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--delta-s", type=float, default=DELTA_S)
    ap.add_argument("--delta-e", type=float, default=DELTA_E)
    ap.add_argument("--schedule-from", type=str, default=None,
                    help="finished bpas run whose per-epoch prune counts the "
                         "matched arms reproduce")
    ap.add_argument("--dsi-target", type=float, default=None,
                    help="fix DSI and vary only the selection criterion, so "
                         "accuracy-vs-DSI curves are comparable point by point")
    ap.add_argument("--acc-ema", type=float, default=0.8,
                    help="per-sample correctness EMA for sample_accuracy")
    ap.add_argument("--keep-floor", type=float, default=0.10,
                    help="lower bound on active batches as a share of n0, for "
                         "the --dsi-target schedule of the matched arms")
    ap.add_argument("--bpas-keep-floor", type=float, default=0.0,
                    help="lower bound on active batches as a share of n0 for "
                         "the bpas arm itself. 0.0 (default) is the published "
                         "behaviour: no floor, so a widened --delta-e can "
                         "clear the active set and end the run before the "
                         "cosine schedule finishes, which confounds data "
                         "removal with truncated annealing. A nonzero value "
                         "keeps the run to full length so the delta_e sweep "
                         "measures data removal alone")
    ap.add_argument("--static-scores", type=str, default=None)
    ap.add_argument("--static-metric", type=str, default=None)
    ap.add_argument("--prune-side", choices=["high", "low"], default="low",
                    help="which end of the metric is pruned first; for "
                         "saturation-time, low = learned quickly = easy")
    ap.add_argument("--batch-composition", choices=["random", "by-metric"],
                    default="random")
    ap.add_argument("--output-path", type=str)
    ap.add_argument("--analyze", nargs="+", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.analyze:
        analyze(args.analyze)
        return
    if args.seed is None or args.arm is None or args.output_path is None:
        ap.error("--seed, --arm and --output-path are required")
    run(args)


if __name__ == "__main__":
    main()