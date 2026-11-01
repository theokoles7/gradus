#!/usr/bin/env python3
"""
scratch_live_band_prune.py

Does pruning the extremes of LIVE per-sample gradient activation change the
curriculum penalty?

Hypothesis, grounded in this project's own mechanism results rather than in a
general "these might combine" intuition:

  The frozen-theta diagnostic found the ordering damage tracks between-batch
  difficulty-mean spread and lag-1 residual correlation. The random-pruning
  sweep found the composition term is what grows when batches become internally
  homogeneous. Trimming both tails compresses the score range, so sorted
  batches differ less from each other and a monotone sweep traverses a narrower
  span. PREDICTION: band pruning shrinks the curriculum penalty; random pruning
  at the same keep rate does not.

  The decisive contrast is therefore
      (asc - shuffled) under BAND pruning   vs   (asc - shuffled) under RANDOM
  pruning at an identical keep-rate trajectory.

Relation to prior work
----------------------
Paul, Ganguli & Dziugaite, "Deep Learning on a Data Diet" (2021) already score
examples by expected loss gradient norm (GraNd) and by the error-vector norm
(EL2N), prune low scores, report that excluding a small subset of the very
highest-scoring examples boosts performance, and tune a keep-BAND with its two
cut points as hyperparameters. Their scores are computed once, early, and then
frozen: one-time selection prior to training. On CIFAR-100 they report pruning
about 25% without accuracy loss.

What is not theirs: scores recomputed DURING training so the band moves with
the model's current state, and an ordering arm. This script is the live version
plus the curriculum decomposition. Expect at best parity with full-data
shuffled (77.53); the realistic positive outcome is the curriculum penalty
shrinking, which is a mechanism confirmation, not a win over shuffling.

Scoring
-------
Per-sample classifier-layer gradient norm, analytic from one forward pass:

    ||g_i|| = ||p_i - y_i|| * sqrt(||h_i||^2 + 1)

for cross-entropy, where h_i is the classifier input. EL2N is the ||p_i - y_i||
factor alone; --score el2n selects it. No backward pass either way. A full pass
over 50,000 unaugmented samples at batch 512 is 98 forward batches against 391
training steps, so --refresh-every 5 costs roughly 5%.

Scoring runs in eval mode, so BatchNorm running statistics are not updated.

Pruning schedule (permanent exclusion)
--------------------------------------
Exclusion is permanent: the active set only shrinks. A fixed per-refresh band
would collapse it geometrically (0.8 per refresh over 20 refreshes leaves ~1%),
so the ACTIVE-SET SIZE follows a schedule and the live scores decide only WHICH
samples go:

    target(e) = N * (1 - (1 - keep) * min(1, e / prune_until))

At each refresh the active set is trimmed to target(e), removing the most
extreme current scores, split between the low and high tails by --tail-split.
After prune_until the set is fixed and refreshes stop changing it.

--prune-mode random removes uniformly at random instead, following the IDENTICAL
size trajectory, so the control differs only in which samples are dropped.

Learning rate
-------------
Cosine is driven by STEP, not epoch, because the active set shrinks and so
batches-per-epoch varies. Total steps are fixed (--fixed-steps, default 39,000,
matching the 100-epoch full-data cell) so every arm gets the same budget. The
existing epoch-driven full-data runs are therefore only approximately
comparable; band and random arms here are exactly comparable to each other.

Usage
-----
    python scratch_live_band_prune.py --self-test

    python scratch_live_band_prune.py --seed 1 --prune-mode band \\
        --arm holistic_asc --keep 0.8 \\
        --output-path results/liveband/band_asc_seed1.json

    python scratch_live_band_prune.py --analyze "results/liveband/*.json"
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
F = None
DataLoader = None
CIFAR100 = None
Compose = Normalize = RandomCrop = RandomHorizontalFlip = ToTensor = None
ResNet18 = None
get_system_core_count = None


def _lazy_imports():
    global torch, nn, F, DataLoader, CIFAR100
    global Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
    global ResNet18, get_system_core_count

    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.utils.data import DataLoader as _DL
    from torchvision.datasets import CIFAR100 as _C
    from torchvision.transforms import (Compose as _Cp, Normalize as _Nm,
                                        RandomCrop as _RC,
                                        RandomHorizontalFlip as _RH,
                                        ToTensor as _TT)
    # RECONCILE: confirm against the installed package layout.
    from gradus.networks.resnet.resnet_18 import ResNet18 as _R
    from gradus.utilities import get_system_core_count as _c

    torch, nn, F, DataLoader, CIFAR100 = (_t, _nn, _F, _DL, _C)
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor = (
        _Cp, _Nm, _RC, _RH, _TT)
    ResNet18, get_system_core_count = _R, _c


BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
NUM_CLASSES = 100
FIXED_STEPS = 39000
EVAL_EVERY_STEPS = 390
SCORE_BS = 512

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"

# NOTE ON ORIENTATION: with --score accuracy the score is EMA correctness, so
# ASCENDING sorts lowest-accuracy (hardest) first. That is the ANTI-curriculum
# direction. Easy-to-hard under an accuracy score is holistic_desc. With
# saturation-time the orientation is reversed (low saturation time = easy), so
# ascending there is easy-first. Always check which direction a given score
# makes "easy" before reading an arm as a curriculum.
ARMS = ("shuffled", "holistic_asc", "holistic_desc", "holistic_bshuf")
MODES = ("band", "random")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets():
    train_tf = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(),
                        ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    eval_tf = Compose([ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    return (CIFAR100(root=DATA_ROOT, train=True, download=True,
                     transform=train_tf),
            CIFAR100(root=DATA_ROOT, train=True, download=True,
                     transform=eval_tf),
            CIFAR100(root=DATA_ROOT, train=False, download=True,
                     transform=eval_tf))


# --------------------------------------------------------------------------
# Pruning schedule and trimming
# --------------------------------------------------------------------------

def target_size_per_epoch(n_active, n_total, rate, floor):
    """Remove `rate` of the CURRENT active set, stopping at floor * n_total.

    Compounding matters: at rate 0.10, 0.9^n leaves 59% of the data after 5
    epochs, 35% after 10, 12% after 20 and 0.5% after 50. Without a floor this
    is not a curriculum experiment, it is a data-shortage experiment whose
    endpoint this project has already measured (-6.78pp at 10% data). The floor
    is what keeps it interpretable.
    """
    return max(int(round(floor * n_total)), int(round((1.0 - rate) * n_active)))


def target_size(epoch, n_total, keep, prune_until):
    """Active-set size at a given epoch. Linear from n_total to keep*n_total
    over prune_until epochs, then constant.
    """
    frac = 1.0 - (1.0 - keep) * min(1.0, epoch / max(prune_until, 1e-9))
    return int(round(frac * n_total))


def trim_band(active, scores, n_keep, tail_split, rng):
    """Remove the most extreme current scores, split between the two tails.

    tail_split is the share of removals taken from the LOW tail (0.5 = symmetric,
    1.0 = low tail only, 0.0 = high tail only).
    """
    n_remove = len(active) - n_keep
    if n_remove <= 0:
        return np.asarray(active), 0, 0
    order = np.argsort(scores, kind="stable")     # ascending score
    n_low = int(round(tail_split * n_remove))
    n_high = n_remove - n_low
    drop = np.concatenate([order[:n_low], order[len(order) - n_high:]
                           if n_high > 0 else order[:0]])
    mask = np.ones(len(active), dtype=bool)
    mask[drop] = False
    return np.asarray(active)[mask], n_low, n_high


def trim_random(active, n_keep, rng):
    """Same size trajectory, uniformly chosen removals."""
    if len(active) <= n_keep:
        return np.asarray(active), 0, 0
    keep_idx = np.sort(rng.choice(len(active), size=n_keep, replace=False))
    return np.asarray(active)[keep_idx], len(active) - n_keep, 0


def epoch_batches(arm, active, scores, bs, rng):
    """Batches over the CURRENT active set. Curriculum arms sort by the current
    live score, so the ordering is as dynamic as the pruning.
    """
    a = np.asarray(active)
    nb = len(a) // bs
    if nb == 0:
        raise SystemExit("active set smaller than one batch")

    if arm == "shuffled":
        o = a[rng.permutation(len(a))]
        return [o[t * bs:(t + 1) * bs] for t in range(nb)]

    srt = a[np.argsort(scores, kind="stable")]
    if arm == "holistic_desc":
        srt = srt[::-1].copy()
    batches = [srt[t * bs:(t + 1) * bs] for t in range(nb)]
    if arm in ("holistic_asc", "holistic_desc"):
        return batches
    if arm == "holistic_bshuf":
        return [batches[i] for i in rng.permutation(nb)]
    raise ValueError(arm)


# --------------------------------------------------------------------------
# Live scoring
# --------------------------------------------------------------------------

class FeatureTap:
    def __init__(self, model):
        self.h = None
        # RECONCILE: Gradus names the classifier _fc_.
        self._handle = model._fc_.register_forward_hook(self._hook)

    def _hook(self, _m, inp, _out):
        self.h = inp[0].detach()

    def close(self):
        self._handle.remove()


def score_samples(model, dataset, indices, device, score_kind, num_workers,
                  acc_state=None, acc_ema=0.7):
    """Per-sample score over `indices`, forward only, unaugmented.

        grad_norm:  ||p - y|| * sqrt(||h||^2 + 1)   exact classifier-layer norm
        el2n:       ||p - y||                       Paul et al.'s EL2N
        accuracy:   EMA of per-epoch correctness, tie-broken by margin

    On `accuracy`: raw per-epoch correctness is BINARY, so after W warmup epochs
    an unsmoothed accuracy score takes at most W+1 distinct values. Sorting
    50,000 samples into 3 or 4 tie groups leaves the order arbitrary inside
    groups of ~12,500, which is most of the ordering. So the score is

        ema_correct  +  1e-3 * margin_percentile

    where ema_correct is an EMA (beta = --acc-ema) of the 0/1 correctness
    updated once per scoring pass, and the margin term is the percentile rank
    of (true logit - best other logit) within this pass. The tiebreak is two
    orders of magnitude below the minimum spacing of the EMA, so correctness
    dominates and margin only orders within ties. Margin's sign is exactly
    correctness, so the tiebreak is the continuous version of the same quantity.

    acc_state is a full-length array carrying ema_correct across passes; it is
    updated in place for the scored indices.

    Eval mode, so BatchNorm running statistics are untouched.
    """
    was = model.training
    model.eval()
    tap = FeatureTap(model)
    out = np.empty(len(indices), dtype=np.float32)
    acc_buf = np.empty(len(indices), dtype=np.float32)
    mar_buf = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(indices), SCORE_BS):
            idx = indices[s:s + SCORE_BS]
            xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
            ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
            logits = model(xs)
            p = torch.softmax(logits.float(), dim=1)
            y1 = torch.zeros_like(p).scatter_(1, ys[:, None], 1.0)
            err = (p - y1).norm(dim=1)
            if score_kind == "el2n":
                v = err
            elif score_kind == "accuracy":
                true_logit = logits.gather(1, ys[:, None]).squeeze(1)
                tmp = logits.clone()
                tmp.scatter_(1, ys[:, None], float("-inf"))
                margin = true_logit - tmp.max(dim=1).values
                correct = (logits.argmax(1) == ys).float()
                v = torch.stack([correct, margin], dim=1)
            else:
                h = tap.h.float()
                v = err * torch.sqrt((h * h).sum(1) + 1.0)
            if score_kind == "accuracy":
                acc_buf[s:s + len(idx)] = v[:, 0].cpu().numpy()
                mar_buf[s:s + len(idx)] = v[:, 1].cpu().numpy()
            else:
                out[s:s + len(idx)] = v.cpu().numpy()
    tap.close()
    if was:
        model.train()

    if score_kind == "accuracy":
        idx_arr = np.asarray(indices)
        prev = acc_state[idx_arr]
        fresh = np.isnan(prev)
        new = np.where(fresh, acc_buf,
                       acc_ema * np.nan_to_num(prev) + (1 - acc_ema) * acc_buf)
        acc_state[idx_arr] = new
        # percentile rank of margin within this pass, in [0, 1)
        r = np.argsort(np.argsort(mar_buf)).astype(np.float64)
        pct = r / max(len(r), 1)
        out = (new + 1e-3 * pct).astype(np.float32)
    return out


def class_entropy(labels, n_classes=NUM_CLASSES):
    c = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    p = c / max(c.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

class IndexBatchLoader:
    def __init__(self, dataset, batches, device):
        self.dataset, self.batches, self.device = dataset, batches, device

    def __iter__(self):
        for idx in self.batches:
            xs = torch.stack([self.dataset[int(i)][0] for i in idx])
            ys = torch.tensor([self.dataset[int(i)][1] for i in idx])
            yield (xs.to(self.device, non_blocking=True),
                   ys.to(self.device, non_blocking=True))


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
    nw = get_system_core_count()
    ds_train, ds_score, ds_test = build_datasets()
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                             num_workers=nw, pin_memory=True)
    targets = np.asarray(ds_train.targets)
    n_total = len(ds_train)

    set_seed(args.seed)
    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    brng = np.random.default_rng(args.seed * 1000 + 7)
    prng = np.random.default_rng(args.seed * 1000 + 31)   # pruning decisions

    active = np.arange(n_total)
    cur_scores = None
    frozen_scores = None          # full-length; set once after warmup
    acc_state = (np.full(n_total, np.nan, dtype=np.float64)
                 if args.score == "accuracy" else None)
    log, refreshes = [], []
    best, steps, epoch = 0.0, 0, 0
    next_eval = args.eval_every_steps

    while steps < args.fixed_steps:
        epoch += 1

        in_warmup = epoch <= args.warmup_epochs

        # ---- refresh: rescore, then trim ------------------------------------
        if in_warmup or (epoch - 1 - args.warmup_epochs) % args.refresh_every == 0:
            cur_scores = score_samples(model, ds_score, active, device,
                                       args.score, nw, acc_state=acc_state,
                                       acc_ema=args.acc_ema)
            if (not in_warmup) and frozen_scores is None:
                # snapshot the first post-warmup score for the frozen arm
                frozen_scores = np.full(n_total, np.nan, dtype=np.float32)
                frozen_scores[active] = cur_scores

            if args.prune_per_epoch > 0:
                tgt = (len(active) if in_warmup
                       else target_size_per_epoch(len(active), n_total,
                                                  args.prune_per_epoch,
                                                  args.keep_floor))
            else:
                tgt = (len(active) if in_warmup
                       else target_size(epoch - 1 - args.warmup_epochs,
                                        n_total, args.keep, args.prune_until))
            before = len(active)
            if args.prune_mode == "band":
                active, n_lo, n_hi = trim_band(active, cur_scores, tgt,
                                               args.tail_split, prng)
            else:
                active, n_lo, n_hi = trim_random(active, tgt, prng)
            if before != len(active):
                cur_scores = score_samples(model, ds_score, active, device,
                                           args.score, nw, acc_state=acc_state,
                                           acc_ema=args.acc_ema)
            refreshes.append({
                "epoch": epoch, "step": steps, "target": tgt,
                "before": before, "after": int(len(active)),
                "removed_low": int(n_lo), "removed_high": int(n_hi),
                "score_min": float(cur_scores.min()),
                "score_med": float(np.median(cur_scores)),
                "score_max": float(cur_scores.max()),
                "class_entropy": class_entropy(targets[active]),
            })
            print(f"  refresh ep {epoch}: {before} -> {len(active)} "
                  f"(low {n_lo}, high {n_hi})  score "
                  f"[{cur_scores.min():.3f}, {cur_scores.max():.3f}]  "
                  f"clsH {refreshes[-1]['class_entropy']:.3f}")

        # Warmup is always shuffled and unpruned. After it, ordering uses
        # either the live score or the frozen post-warmup snapshot; PRUNING is
        # live in both arms, so --resort isolates resorting alone.
        if in_warmup:
            order_arm, order_scores = "shuffled", cur_scores
        elif args.resort == "frozen" and frozen_scores is not None:
            order_arm, order_scores = args.arm, frozen_scores[active]
        else:
            order_arm, order_scores = args.arm, cur_scores
        batches = epoch_batches(order_arm, active, order_scores, BATCH_SIZE,
                                brng)
        loader = IndexBatchLoader(ds_train, batches, device)

        model.train()
        rl, seen = 0.0, 0
        for x, y in loader:
            # step-driven cosine: the active set shrinks, so batches per epoch
            # varies and an epoch-driven schedule would not be comparable
            lr = BASE_LR * (1 + math.cos(math.pi * steps / args.fixed_steps)) / 2
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            rl += loss.item() * y.size(0)
            seen += y.size(0)
            steps += 1
            if steps >= args.fixed_steps:
                break

        if steps >= next_eval or steps >= args.fixed_steps:
            acc = evaluate(model, test_loader, device)
            best = max(best, acc)
            log.append({"epoch": epoch, "step": steps, "lr": lr,
                        "n_active": int(len(active)),
                        "train_loss": rl / max(seen, 1), "test_acc": acc})
            while next_eval <= steps:
                next_eval += args.eval_every_steps
            if len(log) % 10 == 0 or steps >= args.fixed_steps:
                print(f"ep {epoch:4d} step {steps:6d} n={len(active):5d} "
                      f"loss={rl/max(seen,1):.4f} test={acc:.2f} best={best:.2f}")

    out = {"seed": args.seed, "arm": args.arm, "prune_mode": args.prune_mode,
           "resort": args.resort, "warmup_epochs": args.warmup_epochs,
           "prune_per_epoch": args.prune_per_epoch,
           "keep_floor": args.keep_floor, "acc_ema": args.acc_ema,
           "keep": args.keep, "prune_until": args.prune_until,
           "refresh_every": args.refresh_every, "score": args.score,
           "tail_split": args.tail_split, "batch_size": BATCH_SIZE,
           "fixed_steps": args.fixed_steps, "total_steps": steps,
           "epochs": epoch, "n_evals": len(log),
           "final_active": int(len(active)),
           "exclusion": "permanent; samples are never re-admitted",
           "lr_schedule": "cosine driven by step, not epoch",
           "best_test_acc": best, "final_test_acc": log[-1]["test_acc"],
           "refreshes": refreshes, "epochs_log": log}
    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{args.prune_mode}/{args.arm} seed {args.seed}: best={best:.2f} "
          f"final={log[-1]['test_acc']:.2f}  active={len(active)}")
    print(f"-> {args.output_path}")


# --------------------------------------------------------------------------

def self_test():
    N, bs = 50000, 128

    print("per-epoch pruning at rate 0.10 with a floor (compounding):")
    print(f"{'epoch':>6}{'active':>9}{'share':>8}{'batches':>9}")
    for floor in (0.5, 0.0):
        a = N
        traj = []
        for e in range(1, 31):
            a = target_size_per_epoch(a, N, 0.10, floor)
            traj.append((e, a))
        tag = f"floor {floor:.0%}"
        for e, v in traj:
            if e in (1, 3, 5, 7, 10, 20, 30):
                print(f"{e:>6}{v:>9}{v/N:>8.1%}{v // bs:>9}   {tag}")
        print()

    print("accuracy score after W warmup epochs, WITHOUT a tiebreak:")
    rng = np.random.default_rng(0)
    for W in (1, 2, 3):
        acc = rng.binomial(W, 0.35, size=N) / W
        u = len(np.unique(acc))
        big = np.bincount((acc * W).astype(int)).max()
        print(f"  W={W}: {u} distinct values, largest tie group {big} samples "
              f"({big/N:.0%} of the data ordered arbitrarily)")

    print("\nwith the margin-percentile tiebreak:")
    for W in (1, 2, 3):
        acc = rng.binomial(W, 0.35, size=N) / W
        margin = rng.normal(size=N)
        pct = np.argsort(np.argsort(margin)) / N
        score = acc + 1e-3 * pct
        print(f"  W={W}: {len(np.unique(score))} distinct values; "
              f"tiebreak range {1e-3:.0e} vs EMA spacing {1/W:.3f} "
              f"(ratio {1e-3*W:.1e}, so correctness dominates)")

    # the tiebreak must never reorder across correctness levels
    acc = np.array([0.0, 0.0, 1/3, 1/3, 1.0])
    pct = np.array([0.99, 0.01, 0.99, 0.01, 0.5])
    sc = acc + 1e-3 * pct
    order = np.argsort(sc)
    print(f"\n  ordering check: correctness levels stay separated = "
          f"{bool(np.all(np.diff(acc[order]) >= 0))}")

    print("\nsize trajectory for the linear schedule (unchanged path):")
    for e in (0, 10, 30, 50):
        t = target_size(e, N, 0.8, 50)
        print(f"  epoch {e:>3}: {t} ({t/N:.0%})")

    a = np.arange(1000)
    sc2 = np.arange(1000).astype(float)
    for ts in (0.5, 1.0, 0.0):
        kept, lo, hi = trim_band(a, sc2, 900, ts, None)
        print(f"  tail_split {ts}: removed low {lo}, high {hi}, "
              f"kept [{kept.min()}, {kept.max()}]")
    print("\nself-test passed")


def analyze(paths):
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit("no files matched")
    by = defaultdict(dict)
    meta = {}
    for f in files:
        L = json.load(open(f))
        key_mode = L["prune_mode"]
        if L.get("resort") == "frozen":
            key_mode += "/frozen"
        by[(key_mode, L["arm"])][L["seed"]] = L["best_test_acc"]
        meta[L["prune_mode"]] = L
    modes = sorted({k[0] for k in by})
    seeds = sorted(set.intersection(*[set(v) for v in by.values()]))
    print(f"{len(seeds)} shared seed(s): {seeds}; keep="
          f"{list(meta.values())[0]['keep']}, score="
          f"{list(meta.values())[0]['score']}, "
          f"prune_until={list(meta.values())[0]['prune_until']}\n")

    print(f"{'mode':<8}" + "".join(f"{a:>17}" for a in ARMS))
    col = {}
    for m in modes:
        vals = []
        for a in ARMS:
            v = np.array([by[(m, a)][s] for s in seeds]) if (m, a) in by else None
            col[(m, a)] = v
            vals.append("      n/a        " if v is None
                        else f"{v.mean():>11.2f} (sd {v.std(ddof=1):.2f})"
                        if len(seeds) > 1 else f"{v.mean():>17.2f}")
        print(f"{m:<8}" + "".join(vals))

    print("\ncurriculum penalty within each pruning mode:")
    pen = {}
    for m in modes:
        if col[(m, "shuffled")] is None:
            continue
        for a in ("holistic_asc", "holistic_desc", "holistic_bshuf"):
            if col[(m, a)] is None:
                continue
            d = col[(m, a)] - col[(m, "shuffled")]
            pen[(m, a)] = d
            n = len(d)
            extra = (f"  sd={d.std(ddof=1):.2f}  t={d.mean()/(d.std(ddof=1)/math.sqrt(n)):+.2f} (df={n-1})"
                     if n > 1 else "  (single seed)")
            print(f"  {m:<8} {a:<16} - shuffled = {d.mean():+6.2f}pp{extra}")

    if ("band", "holistic_asc") in pen and ("random", "holistic_asc") in pen:
        diff = pen[("band", "holistic_asc")] - pen[("random", "holistic_asc")]
        n = len(diff)
        print(f"\nDECISIVE CONTRAST  (asc penalty under band) minus "
              f"(asc penalty under random)")
        print(f"  {diff.mean():+.2f}pp" +
              (f"  sd={diff.std(ddof=1):.2f}  "
               f"t={diff.mean()/(diff.std(ddof=1)/math.sqrt(n)):+.2f} (df={n-1})  "
               f"positive={int((diff>0).sum())}/{n}" if n > 1 else "  (single seed)"))
        print("  positive means band pruning SHRINKS the curriculum penalty "
              "relative to random pruning, as predicted.")

    print("\nReference: full-data shuffled is 77.53 +- 0.18 (master_record). "
          "Beating it requires pruning to be neutral AND the penalty to turn "
          "positive; nothing in this corpus supports the latter.")
    print("Computed ad hoc by this script; rerun in your own analysis code "
          "before citing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", choices=ARMS, default="shuffled")
    ap.add_argument("--prune-mode", choices=MODES, default="band")
    ap.add_argument("--keep", type=float, default=0.8)
    ap.add_argument("--prune-until", type=int, default=50,
                    help="epoch by which the active set reaches keep*N")
    ap.add_argument("--refresh-every", type=int, default=5)
    ap.add_argument("--score", choices=["grad_norm", "el2n", "accuracy"],
                    default="grad_norm")
    ap.add_argument("--resort", choices=["live", "frozen"], default="live",
                    help="live re-sorts by the current score each refresh; "
                         "frozen orders by the post-warmup snapshot. Pruning "
                         "is live in both, so this isolates resorting.")
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="shuffled, unpruned epochs that seed the score")
    ap.add_argument("--prune-per-epoch", type=float, default=0.0,
                    help="remove this share of the CURRENT active set each "
                         "refresh, down to --keep-floor. 0 uses the linear "
                         "--keep/--prune-until schedule instead.")
    ap.add_argument("--keep-floor", type=float, default=0.5,
                    help="lower bound on active-set size as a share of N")
    ap.add_argument("--acc-ema", type=float, default=0.7)
    ap.add_argument("--tail-split", type=float, default=0.5,
                    help="share of removals taken from the LOW tail; 0.5 is "
                         "symmetric, 1.0 low only, 0.0 high only")
    ap.add_argument("--fixed-steps", type=int, default=FIXED_STEPS)
    ap.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
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
    if args.seed is None or args.output_path is None:
        ap.error("--seed and --output-path are required")
    run(args)


if __name__ == "__main__":
    main()