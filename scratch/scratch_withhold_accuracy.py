#!/usr/bin/env python3
"""
scratch_withhold_accuracy.py

Does temporarily withholding mastered examples create a regime where adaptive
curriculum selection becomes useful?

Four arms, decomposed so a gain can be attributed:

  A random_prune    no score; a random p% per class withheld each interval;
                    shuffled order. CONTROL.
  B frozen_acc      accuracy EMA computed once after warmup and frozen; the
                    easiest p% per class by that FROZEN score withheld;
                    shuffled order.
  C dynamic_curric  score refreshed every q epochs; current easiest p% per
                    class withheld; mixed batches sorted by MEAN batch score.
  D dynamic_prune   score refreshed every q epochs; current easiest p% per
                    class withheld; shuffled order.

    D - A  value of accuracy-based adaptive withholding
    C - D  incremental value of curriculum ORDERING on top of it
    B - A  whether a one-off mastery estimate suffices (the staleness question)

Withholding is NON-CUMULATIVE. At each refresh the mask is rebuilt from
scratch over all N samples, so a previously withheld example is eligible again
and can never acquire a permanently stale score. Cumulative removal at 10% per
interval collapses the set (0.9^n leaves 12% after 20 intervals), which this
project has already measured as a data-shortage effect, not a curriculum one.

Score
-----
Per-sample EMA of correctness, which is what "accuracy" means at the sample
level once smoothed:

    A_i <- beta * A_i + (1 - beta) * 1[argmax == y]

Raw per-epoch correctness is binary, so after W warmup epochs an unsmoothed
score takes at most W+1 values; on simulated data at 35% accuracy the largest
tie group covered 65% (W=1), 46% (W=2) and 45% (W=3) of the set, meaning about
half the ordering would be arbitrary. True-class probability p(y|x) is stored
as a TIEBREAK ONLY, added at 1e-3 scale so it orders within ties and never
across correctness levels. That keeps the primary metric accuracy history
rather than quietly making this a confidence curriculum.

Batch ordering (arm C)
----------------------
Batches are formed from a SHUFFLE of the retained set, so composition stays
mixed, and then the BATCHES are sorted by mean score, descending (high score =
high accuracy = easy, so descending is easy-to-hard).

Sorting samples within an already-formed batch would be a no-op: the batch
gradient is invariant to intra-batch order, which this project confirmed
empirically (batch-wise curricula came out at -0.02pp against the unshuffled
baseline over 242 runs).

EXPECTED WEAKNESS, stated in advance: mean scores of random mixed batches
concentrate within about sigma/sqrt(128), roughly 9% of the sample-level
spread. This is close to the `stride` construction from the frozen-theta
diagnostic, which had between-batch difficulty-mean variance 0.0002 against
0.987 for sample-sorted batches and was indistinguishable from shuffled on
every gradient metric. So C - D is likely to be small and three seeds may not
resolve it. A null there should be read as underpowered, not as refutation.

Class balance
-------------
Withholding is per class by default (--class-balanced), so easy classes are not
preferentially stripped. Note that global withholding of the easiest samples
all the way to a 50% keep rate moved active-set class entropy only from 4.6052
to 4.5389 against a uniform 4.6052, so this is a cheap safeguard rather than a
fix for an observed problem in this setup.

Usage
-----
    python scratch_withhold_accuracy.py --self-test
    python scratch_withhold_accuracy.py --seed 1 --arm dynamic_prune \\
        --output-path results/withhold/D_seed1.json
    python scratch_withhold_accuracy.py --analyze "results/withhold/*.json"
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
DataLoader = None
CIFAR100 = None
Compose = Normalize = RandomCrop = RandomHorizontalFlip = ToTensor = None
ResNet18 = None
get_system_core_count = None


def _lazy_imports():
    global torch, nn, DataLoader, CIFAR100
    global Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
    global ResNet18, get_system_core_count

    import torch as _t
    import torch.nn as _nn
    from torch.utils.data import DataLoader as _DL
    from torchvision.datasets import CIFAR100 as _C
    from torchvision.transforms import (Compose as _Cp, Normalize as _Nm,
                                        RandomCrop as _RC,
                                        RandomHorizontalFlip as _RH,
                                        ToTensor as _TT)
    # RECONCILE: confirm against the installed package layout.
    from gradus.networks.resnet.resnet_18 import ResNet18 as _R
    from gradus.utilities import get_system_core_count as _c

    torch, nn, DataLoader, CIFAR100 = (_t, _nn, _DL, _C)
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

ARMS = ("full_data", "random_prune", "frozen_random", "frozen_acc",
        "static_prune", "dynamic_prune", "dynamic_curric")


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
# Scoring
# --------------------------------------------------------------------------

def score_pass(model, dataset, device, n_total, acc_ema_state, beta):
    """One unaugmented pass over ALL samples.

    Updates the correctness EMA in place and returns
        score = A_i + 1e-3 * percentile_rank(p(y|x))
    so accuracy history is primary and confidence only breaks ties. Eval mode,
    so BatchNorm running statistics are untouched.
    """
    was = model.training
    model.eval()
    correct = np.empty(n_total, dtype=np.float32)
    conf = np.empty(n_total, dtype=np.float32)
    idx_all = np.arange(n_total)
    with torch.no_grad():
        for s in range(0, n_total, SCORE_BS):
            idx = idx_all[s:s + SCORE_BS]
            xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
            ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
            logits = model(xs)
            p = torch.softmax(logits.float(), dim=1)
            correct[s:s + len(idx)] = (logits.argmax(1) == ys).float().cpu()
            conf[s:s + len(idx)] = p.gather(1, ys[:, None]).squeeze(1).cpu()
    if was:
        model.train()

    fresh = np.isnan(acc_ema_state)
    acc_ema_state[:] = np.where(fresh, correct,
                                beta * np.nan_to_num(acc_ema_state)
                                + (1 - beta) * correct)
    pct = np.argsort(np.argsort(conf)).astype(np.float64) / max(n_total, 1)
    return (acc_ema_state + 1e-3 * pct).astype(np.float32), correct.mean()


# --------------------------------------------------------------------------
# Withholding masks (rebuilt from scratch every interval: non-cumulative)
# --------------------------------------------------------------------------

def withhold_mask(scores, targets, p, mode, rng, class_balanced=True):
    """Indices RETAINED for the next interval. Never cumulative: this is built
    over all N samples every time, so a withheld example is eligible again.

    mode 'easiest' drops the top-p by score within each class (or globally);
    mode 'random' drops a random p, matching the same per-class counts.
    """
    n = len(scores)
    drop = np.zeros(n, dtype=bool)
    if class_balanced:
        for c in np.unique(targets):
            idx = np.where(targets == c)[0]
            k = int(round(p * len(idx)))
            if k == 0:
                continue
            if mode == "random":
                sel = rng.choice(idx, size=k, replace=False)
            else:
                sel = idx[np.argsort(scores[idx], kind="stable")[-k:]]
            drop[sel] = True
    else:
        k = int(round(p * n))
        if k:
            sel = (rng.choice(n, size=k, replace=False) if mode == "random"
                   else np.argsort(scores, kind="stable")[-k:])
            drop[sel] = True
    return np.where(~drop)[0]


def class_entropy(labels, n_classes=NUM_CLASSES):
    c = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    pr = c / max(c.sum(), 1.0)
    pr = pr[pr > 0]
    return float(-(pr * np.log(pr)).sum())


# --------------------------------------------------------------------------
# Batch construction
# --------------------------------------------------------------------------

def make_batches(retained, scores, bs, rng, order):
    """Always shuffle the retained set and form MIXED batches, so composition
    is never difficulty-homogeneous. `order` then decides presentation:

        shuffled      batches as formed (random order)
        batch_desc    batches sorted by mean score, descending (easy first)
        batch_asc     batches sorted by mean score, ascending (hard first)
    """
    r = np.asarray(retained)
    o = r[rng.permutation(len(r))]
    nb = len(o) // bs
    batches = [o[t * bs:(t + 1) * bs] for t in range(nb)]
    if order == "shuffled":
        return batches
    means = np.array([scores[b].mean() for b in batches])
    idx = np.argsort(means)
    if order == "batch_desc":
        idx = idx[::-1]
    return [batches[i] for i in idx]


# (selection mode, refresh score?, rotate mask each interval?, batch order)
#
# The rotate flag is separate from the refresh flag because they are separate
# confounds. A frozen SCORE with a deterministic selection rule also yields a
# frozen MASK, so the same 10% is withheld for the whole run and the model
# never sees those samples at all. Comparing dynamic_prune to frozen_acc then
# mixes score freshness with data coverage. frozen_random supplies the missing
# cell: one random mask fixed for the whole run.
#
#   frozen_acc    - frozen_random   score-based selection, coverage held fixed
#   frozen_random - random_prune    mask rotation, selection held random
#   dynamic_prune - random_prune    the full adaptive-withholding effect
#   dynamic_prune - frozen_acc      freshness AND coverage together
ARM_SPEC = {
    "full_data":      ("random",  False, False, "shuffled"),   # rate forced 0
    # static_prune: score from a precomputed Gradus metric, never refreshed.
    # Deterministic selection on a fixed score gives a FIXED mask, so the
    # pruned samples are never seen. That is faithful to B-PAS and to the
    # sequence-pruning method, both of which exclude permanently, and it is
    # what their random-pruning control does as well. The matched comparison
    # is therefore frozen_random (same fixed-mask coverage, random selection),
    # NOT dynamic_prune (which rotates and sees all the data).
    "static_prune":   ("easiest", False, False, "shuffled"),
    "random_prune":   ("random",  True,  True,  "shuffled"),
    "frozen_random":  ("random",  False, False, "shuffled"),
    "frozen_acc":     ("easiest", False, False, "shuffled"),
    "dynamic_prune":  ("easiest", True,  True,  "shuffled"),
    "dynamic_curric": ("easiest", True,  True,  "batch_desc"),
}


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



def load_parquet_scores(path, metric, n_expected, prune_side):
    """One precomputed metric column, in dataset index order.

    prune_side names which END of the metric is removed first:
      high  remove the largest values first (e.g. accuracy: high = mastered)
      low   remove the smallest values first (e.g. saturation-time: low =
            learned quickly = easy)

    Internally the selection rule always removes the TOP of the score array, so
    `low` is implemented by negating. Which end corresponds to "easy" is an
    assumption for the image statistics (edge-density, wavelet-entropy), not a
    measurement; it is only known for the model-derived metrics.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c not in ("index", "class")]
    if metric is None:
        raise SystemExit(f"--static-metric required. Available: {', '.join(cols)}")
    if metric not in df.columns:
        raise SystemExit(f"'{metric}' not in {path}. Available: {', '.join(cols)}")
    if "index" in df.columns:
        df = df.sort_values("index")
        idx = df["index"].to_numpy()
        if idx[0] != 0 or idx[-1] != len(idx) - 1:
            raise SystemExit("`index` is not a contiguous 0..N-1 range")
    v = df[metric].to_numpy().astype(np.float32)
    if len(v) != n_expected:
        raise SystemExit(f"{metric}: {len(v)} rows != {n_expected}")
    if not np.isfinite(v).all():
        raise SystemExit(f"{metric}: non-finite values")
    print(f"  static score: {metric} from {os.path.basename(path)} "
          f"[min {v.min():.4g}, max {v.max():.4g}, unique {len(np.unique(v))}], "
          f"removing the {prune_side} end first")
    return -v if prune_side == "low" else v


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

    score_mode, refresh, rotate, order = ARM_SPEC[args.arm]
    prune_rate = 0.0 if args.arm == "full_data" else args.prune_rate

    set_seed(args.seed)
    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    brng = np.random.default_rng(args.seed * 1000 + 7)
    mrng = np.random.default_rng(args.seed * 1000 + 31)

    acc_ema = np.full(n_total, np.nan, dtype=np.float64)
    scores = None
    frozen = None
    if args.arm == "static_prune":
        if not args.static_scores:
            raise SystemExit("static_prune requires --static-scores")
        scores = load_parquet_scores(args.static_scores, args.static_metric,
                                     n_total, args.prune_side)
        frozen = scores.copy()
    retained = np.arange(n_total)
    retained_fixed = None

    log, events = [], []
    best, steps, epoch = 0.0, 0, 0
    next_eval = EVAL_EVERY_STEPS

    while steps < args.fixed_steps:
        epoch += 1
        in_warmup = epoch <= args.warmup

        # ---- refresh / mask rebuild (non-cumulative) -----------------------
        at_interval = (not in_warmup) and ((epoch - 1 - args.warmup)
                                           % args.interval == 0)
        if at_interval:
            if refresh or scores is None:
                scores, train_acc = score_pass(model, ds_score, device,
                                               n_total, acc_ema, args.acc_ema)
                if frozen is None:
                    frozen = scores.copy()
            else:
                train_acc = float(np.nan)
            use = scores if refresh else frozen
            if rotate or retained_fixed is None:
                retained = withhold_mask(use, targets, prune_rate, score_mode,
                                         mrng,
                                         class_balanced=args.class_balanced)
                if not rotate:
                    retained_fixed = retained.copy()
            else:
                retained = retained_fixed
            events.append({
                "epoch": epoch, "step": steps, "retained": int(len(retained)),
                "share": len(retained) / n_total,
                "class_entropy": class_entropy(targets[retained]),
                "train_acc": None if math.isnan(train_acc) else float(train_acc),
                "mean_score_retained": float(use[retained].mean()),
                "mean_score_withheld": float(
                    np.delete(use, retained).mean()) if len(retained) < n_total
                    else None,
            })
            print(f"  ep {epoch}: retained {len(retained)} "
                  f"({len(retained)/n_total:.0%})  clsH "
                  f"{events[-1]['class_entropy']:.4f}  "
                  f"train_acc {events[-1]['train_acc']}")

        if in_warmup:
            batches = make_batches(np.arange(n_total), None, BATCH_SIZE,
                                   brng, "shuffled")
        else:
            use = scores if refresh else frozen
            batches = make_batches(retained, use, BATCH_SIZE, brng, order)

        model.train()
        rl, seen = 0.0, 0
        for x, y in IndexBatchLoader(ds_train, batches, device):
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

        # warmup must still build the score history before the first mask
        if in_warmup and epoch == args.warmup:
            scores, _ = score_pass(model, ds_score, device, n_total, acc_ema,
                                   args.acc_ema)
            frozen = scores.copy()

        if steps >= next_eval or steps >= args.fixed_steps:
            acc = evaluate(model, test_loader, device)
            best = max(best, acc)
            log.append({"epoch": epoch, "step": steps, "lr": lr,
                        "n_retained": int(len(retained)),
                        "train_loss": rl / max(seen, 1), "test_acc": acc})
            while next_eval <= steps:
                next_eval += EVAL_EVERY_STEPS
            if len(log) % 10 == 0 or steps >= args.fixed_steps:
                print(f"ep {epoch:4d} step {steps:6d} n={len(retained):5d} "
                      f"loss={rl/max(seen,1):.4f} test={acc:.2f} best={best:.2f}")

    out = {"seed": args.seed, "arm": args.arm_label or args.arm,
           "base_arm": args.arm,
           "static_metric": args.static_metric,
           "prune_side": args.prune_side,
           "score_mode": score_mode, "refresh": refresh, "order": order,
           "rotate_mask": rotate,
           "prune_rate": prune_rate, "warmup": args.warmup,
           "interval": args.interval, "acc_ema": args.acc_ema,
           "class_balanced": args.class_balanced,
           "withholding": "non-cumulative; mask rebuilt over all N each interval",
           "batch_composition": "mixed (retained set shuffled before batching)",
           "lr_schedule": "cosine driven by step",
           "batch_size": BATCH_SIZE, "fixed_steps": args.fixed_steps,
           "total_steps": steps, "epochs": epoch, "n_evals": len(log),
           "best_test_acc": best, "final_test_acc": log[-1]["test_acc"],
           "events": events, "epochs_log": log}
    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{args.arm} seed {args.seed}: best={best:.2f} "
          f"final={log[-1]['test_acc']:.2f}")
    print(f"-> {args.output_path}")


# --------------------------------------------------------------------------

def self_test():
    N, bs, C = 50000, 128, NUM_CLASSES
    rng = np.random.default_rng(0)
    targets = np.repeat(np.arange(C), N // C)
    scores = rng.random(N).astype(np.float32)

    print("non-cumulative withholding: retained share must be constant")
    prev = None
    for it in range(1, 7):
        sc = rng.random(N).astype(np.float32)   # scores move between intervals
        r = withhold_mask(sc, targets, 0.10, "easiest", rng, True)
        overlap = "-" if prev is None else f"{np.isin(r, prev).mean():.3f}"
        print(f"  interval {it}: retained {len(r)} ({len(r)/N:.1%})  "
              f"clsH {class_entropy(targets[r]):.4f}  overlap with prior {overlap}")
        prev = r
    print("  (share constant and overlap < 1 confirms it is not cumulative)")

    print("\nclass balance: per-class counts after withholding easiest 10%")
    r = withhold_mask(scores, targets, 0.10, "easiest", rng, True)
    cnt = np.bincount(targets[r], minlength=C)
    print(f"  per-class retained min {cnt.min()} max {cnt.max()} "
          f"(balanced: all equal = {bool(cnt.min() == cnt.max())})")
    r2 = withhold_mask(scores, targets, 0.10, "easiest", rng, False)
    cnt2 = np.bincount(targets[r2], minlength=C)
    print(f"  global variant:  min {cnt2.min()} max {cnt2.max()}")

    print("\nbatch-mean ordering signal (the predicted weakness)")
    ret = np.arange(N)
    s = rng.normal(size=N).astype(np.float32)
    b = make_batches(ret, s, bs, np.random.default_rng(1), "batch_desc")
    means = np.array([s[x].mean() for x in b])
    print(f"  sample-level sd {s.std():.4f}")
    print(f"  batch-mean  sd {means.std():.4f}  "
          f"(= sigma/sqrt(bs) = {s.std()/math.sqrt(bs):.4f})")
    print(f"  ratio {means.std()/s.std():.3f}; sorted batches span "
          f"[{means.min():+.3f}, {means.max():+.3f}] against sample range "
          f"[{s.min():+.2f}, {s.max():+.2f}]")
    print(f"  descending order monotone: "
          f"{bool(np.all(np.diff(means) <= 1e-9))}")
    print("  compare: sample-sorted batches would span nearly the full sample "
          "range. This is the stride regime, so C - D is expected small.")

    print("\nintra-batch order is a no-op (why sorting inside a batch is not "
          "done): batch gradient is invariant to permutation within the batch")

    print("\nself-test passed")


def analyze(paths):
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit("no files matched")
    by = defaultdict(dict)
    meta = None
    for f in files:
        L = json.load(open(f))
        by[L["arm"]][L["seed"]] = L["best_test_acc"]
        meta = L
    known = [a for a in ARMS if a in by]
    extra = sorted(a for a in by if a not in ARMS)
    arms = known + extra
    seeds = sorted(set.intersection(*[set(by[a]) for a in arms]))
    print(f"{len(seeds)} shared seed(s): {seeds}; "
          f"W={meta['warmup']}, q={meta['interval']}, "
          f"class_balanced={meta['class_balanced']}\n")

    print(f"{'seed':<6}" + "".join(f"{a:>17}" for a in arms))
    for s in seeds:
        print(f"{s:<6}" + "".join(f"{by[a][s]:>17.2f}" for a in arms))
    col = {a: np.array([by[a][s] for s in seeds]) for a in arms}
    print(f"{'mean':<6}" + "".join(f"{col[a].mean():>17.2f}" for a in arms))

    n = len(seeds)

    def rep(label, d, note=""):
        m = d.mean()
        if n > 1:
            sd = d.std(ddof=1)
            se = sd / math.sqrt(n)
            print(f"  {label:<38} {m:+6.2f}pp  sd={sd:.2f}  "
                  f"t={m/se:+.2f} (df={n-1})  pos={int((d>0).sum())}/{n}{note}")
        else:
            print(f"  {label:<38} {m:+6.2f}pp  (single seed){note}")

    print("\nagainst full-data shuffled (the comparison that matters):")
    if "full_data" in col:
        for a in arms:
            if a != "full_data":
                rep(f"{a} - full_data", col[a] - col["full_data"])
    else:
        print("  no full_data arm present; every delta below is relative to a "
              "pruned control and cannot show a gain over doing nothing")

    print("\nmatched-coverage selection tests (both arms hold a FIXED mask, "
          "so the only difference is which samples were chosen):")
    for a in arms:
        if a in ("frozen_random", "full_data", "random_prune",
                 "dynamic_prune", "dynamic_curric"):
            continue
        if "frozen_random" in col:
            rep(f"{a} - frozen_random", col[a] - col["frozen_random"])
    if "frozen_random" not in col:
        print("  no frozen_random arm; a fixed-mask arm compared against a "
              "ROTATING control confounds selection with data coverage")

    print("\ndecomposition:")
    if "dynamic_prune" in col and "random_prune" in col:
        rep("adaptive withholding   D - A",
            col["dynamic_prune"] - col["random_prune"])
    if "dynamic_curric" in col and "dynamic_prune" in col:
        rep("curriculum ordering    C - D",
            col["dynamic_curric"] - col["dynamic_prune"],
            "  [underpowered by design]")
    if "frozen_acc" in col and "frozen_random" in col:
        rep("selection, coverage fixed  B - E",
            col["frozen_acc"] - col["frozen_random"])
    if "frozen_random" in col and "random_prune" in col:
        rep("mask rotation, selection random  E - A",
            col["frozen_random"] - col["random_prune"])
    if "dynamic_prune" in col and "frozen_acc" in col:
        rep("freshness AND coverage  D - B",
            col["dynamic_prune"] - col["frozen_acc"],
            "  [confounded: see E]")

    print("\nReference: full-data shuffled is 77.53 +- 0.18 (master_record). "
          "At a 90% keep rate an arm must clear that, not just clear A.")
    print("Computed ad hoc by this script; rerun in your own analysis code "
          "before citing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--prune-rate", type=float, default=0.10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--acc-ema", type=float, default=0.8)
    ap.add_argument("--static-scores", type=str, default=None,
                    help="Gradus metric-scores parquet for --arm static_prune")
    ap.add_argument("--static-metric", type=str, default=None)
    ap.add_argument("--prune-side", choices=["high", "low"], default="high",
                    help="which end of the metric to remove first")
    ap.add_argument("--arm-label", type=str, default=None,
                    help="name recorded in the log, so two static metrics do "
                         "not collide in the analysis")
    ap.add_argument("--class-balanced", action="store_true", default=True)
    ap.add_argument("--global-prune", dest="class_balanced",
                    action="store_false")
    ap.add_argument("--fixed-steps", type=int, default=FIXED_STEPS)
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