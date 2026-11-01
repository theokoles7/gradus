#!/usr/bin/env python3
"""
scratch_ordering_discriminator.py

Does batch ORDERING compound over training, or is the holistic penalty driven
by something else (composition, coverage over time)?

The frozen-theta diagnostic measured gradient noise at single model states and
found, under the ACTUAL original metric (saturation-time), effects far too
small to explain the corpus penalty of -6.09pp for holistic curricula. Either
a small per-step bias compounds over a full trajectory -- which no single-theta
measurement can see -- or the damage is not in gradient geometry at all.

This script trains. Three arms, matched seeds, 20 epochs:

  shuffled         uniform random batches, re-drawn every epoch. Reference.
  holistic_asc     batches built by sorting on the difficulty metric, presented
                   easiest-first. The original holistic curriculum.
  holistic_bshuf   IDENTICAL batch contents to holistic_asc; only the order in
                   which those batches are presented is randomised, re-drawn
                   every epoch.

The comparison that decides it is asc vs bshuf. They see exactly the same
batches, the same number of times, with identical per-epoch sample coverage.
The only difference is sequence.

  asc clearly below bshuf        -> ordering compounds over training; the
                                    frozen-theta measurement was too local.
  asc ~= bshuf, both below shuffled -> ordering is not the mechanism. The
                                    damage is composition or coverage, and the
                                    gradient-noise line should be closed out.
  all three comparable           -> 20 epochs is too short, or the penalty does
                                    not reproduce in this setup at all, which
                                    would itself need explaining before any
                                    further mechanism work.

Why 20 epochs: the corpus penalty is large (-6.09pp mean, worst -25pp). An
effect that size should be visible well before convergence. If it is not
visible by epoch 20, that is informative rather than inconclusive.

Notes on the design
-------------------
* Batches are rebuilt EVERY epoch. For holistic_asc the sort is deterministic,
  so batch contents are identical each epoch (this is what the original
  curriculum did). For holistic_bshuf the contents are the same but the
  presentation order is redrawn, so ordering is randomised independently each
  epoch rather than fixed once.
* Every arm sees exactly the same sample universe: floor(N/bs)*bs samples,
  chosen once per seed, so the remainder never correlates with difficulty.
* Difficulty comes from a Gradus metric-scores parquet. Loss-derived difficulty
  is deliberately NOT the default here -- the phenomenon being explained is the
  original static-metric curricula.
* BatchNorm is in ordinary train mode throughout. This is real training, not a
  frozen-theta probe, so no BN snapshotting.
* Augmentation is ON, matching the original runs. The frozen-theta diagnostic
  turned it off to isolate composition; here we want the real regime.

Random pruning (simulated data shortage)
----------------------------------------
--prune-fraction f keeps a random fraction f of the training set, chosen once
per seed and identical across all arms, so shortage is never confounded with
selection. Metric-based pruning is a different hypothesis (which data, not
which order) and is deliberately not implemented here.

--fixed-steps S holds the number of optimizer steps constant across prune
fractions (more epochs on less data). Without it, pruning at fixed epochs also
cuts compute, which confounds data shortage with a reduced training budget.
The default S = 39,000 matches the existing 100% cell exactly (390 batches x
100 epochs).

Evaluation runs on a fixed STEP cadence (--eval-every-steps, default 390), so
every cell gets ~100 evaluations regardless of prune fraction. Evaluating every
epoch would give the 10% cell ~1,000 evaluations, and best_test_acc, a max over
evaluations, would be inflated by selection alone.

Pre-registered reading of the sweep
-----------------------------------
The mechanism result predicts the holistic penalty SHRINKS as data shrinks, for
reasons unrelated to curriculum helping: drift accumulates over a monotone
sweep through the sorted batches, and a sweep is 390 batches at full data but
39 at 10%. So:

  asc - shuffled moving from ~-3.5pp toward 0   predicted by sweep length alone;
                                                NOT evidence of a curriculum
                                                benefit under shortage
  asc - shuffled POSITIVE, consistently         what the shortage hypothesis
                                                requires
  asc - bshuf shrinking while bshuf - shuffled  confirms the sweep-length
  holds                                         account directly

Usage
-----
    python scratch_ordering_discriminator.py --seed 1 --prune-fraction 0.2 \\
        --fixed-steps 39000 \\
        --static-scores .cache/scores/metric-scores_seed-1.parquet \\
        --static-metric saturation-time \\
        --output-path results/pruning/prune20_seed1.json

    python scratch_ordering_discriminator.py --sweep results/pruning/*.json \\
        results/ordering/ord_100ep_seed*.json

    python scratch_ordering_discriminator.py --seed 1 \\
        --static-scores .cache/scores/metric-scores_seed-1.parquet \\
        --static-metric saturation-time \\
        --output-path results/ordering/ord_seed1.json

    python scratch_ordering_discriminator.py --analyze \\
        results/ordering/ord_seed1.json results/ordering/ord_seed2.json ...
"""

import argparse
import json
import math
import os
import random

import numpy as np

torch = None
nn = None
DataLoader = None
Dataset = None
CosineAnnealingLR = None
CIFAR100 = None
Compose = Normalize = RandomCrop = RandomHorizontalFlip = ToTensor = None
ResNet18 = None
get_system_core_count = None


def _lazy_imports():
    global torch, nn, DataLoader, Dataset, CosineAnnealingLR, CIFAR100
    global Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
    global ResNet18, get_system_core_count

    import torch as _t
    import torch.nn as _nn
    from torch.utils.data import DataLoader as _DL, Dataset as _DS
    from torch.optim.lr_scheduler import CosineAnnealingLR as _Cos
    from torchvision.datasets import CIFAR100 as _C
    from torchvision.transforms import (Compose as _Cp, Normalize as _Nm,
                                        RandomCrop as _RC,
                                        RandomHorizontalFlip as _RH,
                                        ToTensor as _TT)
    # RECONCILE: confirm against the installed package layout.
    from gradus.networks.resnet.resnet_18 import ResNet18 as _R
    from gradus.utilities import get_system_core_count as _c

    torch, nn, DataLoader, Dataset, CosineAnnealingLR, CIFAR100 = (
        _t, _nn, _DL, _DS, _Cos, _C)
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor = (
        _Cp, _Nm, _RC, _RH, _TT)
    ResNet18, get_system_core_count = _R, _c


BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
NUM_CLASSES = 100
EPOCHS = 20

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"

ARMS = ("shuffled", "holistic_asc", "holistic_bshuf")


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
            CIFAR100(root=DATA_ROOT, train=False, download=True,
                     transform=eval_tf))


def load_static_scores(path, metric, n_expected):
    """One metric column from a Gradus metric-scores parquet, in dataset index
    order. Schema: int64 `index`, `class`, one float64 column per metric.
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
            raise SystemExit("`index` is not a contiguous 0..N-1 range; "
                             "cannot align to dataset order")
    v = df[metric].to_numpy().astype(np.float32)
    if len(v) != n_expected:
        raise SystemExit(f"{metric}: {len(v)} rows != {n_expected} samples")
    if not np.isfinite(v).all():
        raise SystemExit(f"{metric}: {int((~np.isfinite(v)).sum())} non-finite")
    ties = len(v) - len(np.unique(v))
    print(f"  difficulty: {metric} [min {v.min():.4g}, max {v.max():.4g}, "
          f"unique {len(np.unique(v))}, tied values {ties}]")
    return v


def choose_universe(n_total, bs, rng, prune_fraction=1.0, prune_rng=None):
    """The sample set every arm uses, chosen once per seed.

    Random pruning first (if prune_fraction < 1), from its own RNG stream so
    the pruned subset does not depend on anything else; then trim to a whole
    number of batches, also at random, so the dropped remainder never
    correlates with difficulty.
    """
    if prune_fraction >= 1.0:
        # unchanged path: reproduces the existing 100% cell exactly
        nb = n_total // bs
        return np.sort(rng.choice(n_total, size=nb * bs, replace=False))

    # Pruned subsets are PREFIXES of one random permutation, so at a given seed
    # the 10% set is contained in the 20% set, which is contained in the 50%
    # set. Differences between prune fractions therefore cannot come from
    # drawing different random samples. The remainder is dropped from the tail
    # of the same random permutation, so it still never correlates with
    # difficulty.
    perm = (prune_rng or rng).permutation(n_total)
    k = int(round(prune_fraction * n_total))
    nb = k // bs
    return np.sort(perm[:nb * bs])


def epoch_batches(arm, universe, scores, bs, rng):
    """Batches for one epoch.

    holistic_asc and holistic_bshuf return the SAME set of batches; only the
    presentation order differs. Verified by --self-test.
    """
    u = np.asarray(universe)
    nb = len(u) // bs

    if arm == "shuffled":
        o = u[rng.permutation(len(u))]
        return [o[t * bs:(t + 1) * bs] for t in range(nb)]

    srt = u[np.argsort(scores[u], kind="stable")]
    batches = [srt[t * bs:(t + 1) * bs] for t in range(nb)]

    if arm == "holistic_asc":
        return batches
    if arm == "holistic_bshuf":
        return [batches[i] for i in rng.permutation(nb)]
    raise ValueError(arm)


class IndexBatchLoader:
    """Yields (x, y) for an explicit list of index arrays, with augmentation
    applied by the dataset as usual. Single-process so batch order is exactly
    what was asked for.
    """

    def __init__(self, dataset, batches, device):
        self.dataset = dataset
        self.batches = batches
        self.device = device

    def __iter__(self):
        for idx in self.batches:
            xs = torch.stack([self.dataset[int(i)][0] for i in idx])
            ys = torch.tensor([self.dataset[int(i)][1] for i in idx])
            yield xs.to(self.device, non_blocking=True), \
                ys.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.batches)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_arm(arm, seed, universe, scores, ds_train, test_loader, device,
              epochs, eval_every_steps):
    """One arm, from scratch. Identical initialisation across arms at a given
    seed, because set_seed is called immediately before the model is built.

    Evaluation happens whenever the cumulative step count crosses a multiple of
    eval_every_steps, and once at the end, so the number of evaluations depends
    on total steps, not on how many epochs those steps are spread over.
    """
    set_seed(seed)
    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    brng = np.random.default_rng(seed * 1000 + 7)

    log = []
    best = 0.0
    steps = 0
    next_eval = eval_every_steps
    for epoch in range(1, epochs + 1):
        batches = epoch_batches(arm, universe, scores, BATCH_SIZE, brng)
        loader = IndexBatchLoader(ds_train, batches, device)

        model.train()
        rl, seen = 0.0, 0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            rl += loss.item() * y.size(0)
            seen += y.size(0)
            steps += 1

        is_last = epoch == epochs
        if steps >= next_eval or is_last:
            acc = evaluate(model, test_loader, device)
            best = max(best, acc)
            log.append({"epoch": epoch, "step": steps,
                        "lr": opt.param_groups[0]["lr"],
                        "train_loss": rl / seen, "test_acc": acc})
            while next_eval <= steps:
                next_eval += eval_every_steps
            if len(log) % 10 == 0 or is_last:
                print(f"    ep {epoch:4d} step {steps:6d} loss={rl/seen:.4f} "
                      f"test={acc:.2f} best={best:.2f}")
        sched.step()

    return {"arm": arm, "best_test_acc": best,
            "final_test_acc": log[-1]["test_acc"], "n_evals": len(log),
            "total_steps": steps, "epochs_log": log}


def self_test():
    """Verify the batching invariants without touching torch."""
    N, bs = 50000, 128
    rng = np.random.default_rng(0)
    scores = rng.normal(size=N).astype(np.float32)
    U = choose_universe(N, bs, np.random.default_rng(3))
    print(f"universe {len(U)} samples = {len(U)//bs} batches")

    B = {a: epoch_batches(a, U, scores, bs, np.random.default_rng(11))
         for a in ARMS}
    ref = np.sort(np.concatenate(B["shuffled"]))
    for a in ARMS:
        f = np.sort(np.concatenate(B[a]))
        print(f"  {a:<16} nb={len(B[a])} same_universe={np.array_equal(f, ref)}")

    asc = sorted(tuple(sorted(x.tolist())) for x in B["holistic_asc"])
    bsh = sorted(tuple(sorted(x.tolist())) for x in B["holistic_bshuf"])
    same_contents = asc == bsh
    same_order = ([x.tolist() for x in B["holistic_asc"]]
                  == [x.tolist() for x in B["holistic_bshuf"]])
    print(f"  asc vs bshuf: identical batch multiset={same_contents}  "
          f"identical order={same_order}")
    assert same_contents and not same_order

    print("\n  per-epoch difficulty-mean sequence (first 8 batches):")
    for a in ARMS:
        m = [float(scores[i].mean()) for i in B[a][:8]]
        print(f"    {a:<16} " + " ".join(f"{v:+.2f}" for v in m))
    print("\nself-test passed")


def run(args):
    _lazy_imports()
    if os.path.exists(args.output_path):
        raise SystemExit(f"{args.output_path} exists. Use a distinct path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = get_system_core_count()
    ds_train, ds_test = build_datasets()
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                             num_workers=nw, pin_memory=True)

    scores = load_static_scores(args.static_scores, args.static_metric,
                                len(ds_train))
    universe = choose_universe(
        len(ds_train), BATCH_SIZE, np.random.default_rng(args.seed + 991),
        prune_fraction=args.prune_fraction,
        prune_rng=np.random.default_rng(args.seed + 5003))
    nb = len(universe) // BATCH_SIZE
    if args.fixed_steps:
        epochs = math.ceil(args.fixed_steps / nb)
    else:
        epochs = args.epochs
    print(f"  prune_fraction={args.prune_fraction}  samples={len(universe)}  "
          f"batches/epoch={nb}  epochs={epochs}  steps~{epochs * nb}")

    out = {"seed": args.seed, "epochs": epochs, "batch_size": BATCH_SIZE,
           "prune_fraction": args.prune_fraction,
           "fixed_steps": args.fixed_steps,
           "batches_per_epoch": nb,
           "eval_every_steps": args.eval_every_steps,
           "pruning": "random, identical across arms at a given seed",
           "arms": list(ARMS), "universe_size": int(len(universe)),
           "difficulty_source": f"{args.static_metric} from "
                                f"{os.path.basename(args.static_scores)}",
           "augmentation": "standard (RandomCrop+Flip), matching the original runs",
           "bn": "ordinary train mode", "results": {}}

    for arm in ARMS:
        print(f"\n=== arm: {arm} (seed {args.seed}) ===")
        out["results"][arm] = train_arm(arm, args.seed, universe, scores,
                                        ds_train, test_loader, device,
                                        epochs, args.eval_every_steps)

    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)

    r = out["results"]
    print(f"\n{'arm':<16}{'best':>8}{'final':>8}")
    for a in ARMS:
        print(f"{a:<16}{r[a]['best_test_acc']:>8.2f}{r[a]['final_test_acc']:>8.2f}")
    print(f"\n-> {args.output_path}")


def analyze(paths):
    logs = [json.load(open(p)) for p in paths]
    print(f"{len(logs)} seed(s); difficulty: {logs[0]['difficulty_source']}; "
          f"{logs[0]['epochs']} epochs\n")

    print(f"{'seed':<6}" + "".join(f"{a:>17}" for a in ARMS))
    for L in logs:
        print(f"{L['seed']:<6}" +
              "".join(f"{L['results'][a]['best_test_acc']:>17.2f}" for a in ARMS))

    # Per-epoch gap trajectory: is the ordering gap still open at the end, or
    # closing? A gap that narrows monotonically late may vanish at convergence.
    ep = [e["epoch"] for e in logs[0]["results"]["shuffled"]["epochs_log"]]
    gap = np.array([[L["results"]["holistic_asc"]["epochs_log"][i]["test_acc"]
                     - L["results"]["holistic_bshuf"]["epochs_log"][i]["test_acc"]
                     for i in range(len(ep))] for L in logs])
    print("\nordering gap (asc - bshuf) by epoch, mean over seeds:")
    step = max(1, len(ep) // 10)
    print("  " + "".join(f"{ep[i]:>7}" for i in range(0, len(ep), step)))
    print("  " + "".join(f"{gap[:, i].mean():>+7.2f}"
                         for i in range(0, len(ep), step)))
    last = gap[:, -5:].mean(axis=0)
    print(f"  last 5 epochs: {' '.join(f'{v:+.2f}' for v in last)}"
          f"   trend {'closing' if last[-1] > last[0] else 'stable or widening'}")

    def col(a):
        return np.array([L["results"][a]["best_test_acc"] for L in logs])

    sh, asc, bsh = col("shuffled"), col("holistic_asc"), col("holistic_bshuf")
    n = len(logs)

    def rep(name, d):
        m = d.mean()
        s = d.std(ddof=1) if n > 1 else float("nan")
        se = s / math.sqrt(n) if n > 1 else float("nan")
        print(f"  {name:<34} mean={m:+7.3f}pp" +
              (f"  sd={s:.3f}  t={m/se:+.2f} (df={n-1})" if n > 1 else
               "  (single seed; no dispersion)"))

    print("\npaired deltas:")
    rep("holistic_asc   - shuffled", asc - sh)
    rep("holistic_bshuf - shuffled", bsh - sh)
    rep("holistic_asc   - holistic_bshuf  <- ORDERING", asc - bsh)

    print("\nverdict:")
    order_gap = float((asc - bsh).mean())
    comp_gap = float((bsh - sh).mean())
    if abs(order_gap) > 1.0 and order_gap < 0:
        print("  asc clearly below bshuf: ORDERING compounds over training. "
              "The frozen-theta measurement was too local.")
    elif abs(order_gap) <= 1.0 and comp_gap < -1.0:
        print("  asc ~= bshuf, both below shuffled: ordering is NOT the "
              "mechanism. Damage is composition or coverage; close out the "
              "gradient-noise line.")
    elif abs(comp_gap) <= 1.0 and abs(order_gap) <= 1.0:
        print("  all arms comparable: the holistic penalty did not reproduce "
              "at this horizon. Either 20 epochs is too short, or the original "
              "-6.09pp figure needs re-examining before more mechanism work.")
    else:
        print(f"  ordering gap {order_gap:+.2f}pp, composition gap "
              f"{comp_gap:+.2f}pp; read the table directly.")
    print("\nThresholds here are descriptive, not tests. These statistics are "
          "computed ad hoc by this script; rerun in your own analysis code "
          "before citing.")



def analyze_sweep(paths):
    """Group runs by prune fraction and report the three decomposed deltas per
    fraction, with the pre-registered reading.
    """
    import glob as _g
    files = []
    for p in paths:
        files.extend(sorted(_g.glob(p)))
    groups = {}
    for f in files:
        L = json.load(open(f))
        frac = float(L.get("prune_fraction", 1.0))
        groups.setdefault(frac, []).append(L)

    print(f"{'frac':>6}{'n':>4}{'batches':>9}{'epochs':>8}"
          f"{'asc-shuf':>18}{'bshuf-shuf':>18}{'asc-bshuf':>18}")
    rows = []
    for frac in sorted(groups, reverse=True):
        logs = groups[frac]
        n = len(logs)
        col = lambda a: np.array([L["results"][a]["best_test_acc"] for L in logs])
        sh, asc, bsh = col("shuffled"), col("holistic_asc"), col("holistic_bshuf")

        def cell(d):
            m = d.mean()
            if n > 1:
                return f"{m:+6.2f} (sd {d.std(ddof=1):.2f})"
            return f"{m:+6.2f} (n=1)   "

        nb = logs[0].get("batches_per_epoch",
                         logs[0].get("universe_size", 0) // BATCH_SIZE)
        print(f"{frac:>6.2f}{n:>4}{nb:>9}{logs[0]['epochs']:>8}"
              f"{cell(asc - sh):>18}{cell(bsh - sh):>18}{cell(asc - bsh):>18}")
        rows.append((frac, float((asc - sh).mean()),
                     int(((asc - sh) > 0).sum()), n))

    print("\npre-registered reading:")
    for frac, m, pos, n in rows:
        if m > 0 and pos == n:
            v = "asc ABOVE shuffled in every seed: candidate positive effect"
        elif m > 0:
            v = f"asc above shuffled on average but only {pos}/{n} seeds"
        else:
            v = "asc below shuffled (penalty shrinking toward 0 is predicted " \
                "by sweep length alone and is NOT a positive result)"
        print(f"  frac {frac:.2f}: {v}")
    print("\nComputed ad hoc by this script; rerun in your own analysis code "
          "before citing. Cells with n<3 have no usable dispersion.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--output-path", type=str)
    ap.add_argument("--static-scores", type=str)
    ap.add_argument("--static-metric", type=str)
    ap.add_argument("--analyze", nargs="+", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prune-fraction", type=float, default=1.0,
                    help="keep this random fraction of the training set, "
                         "identical across arms at a given seed")
    ap.add_argument("--fixed-steps", type=int, default=None,
                    help="hold total optimizer steps fixed (epochs derived). "
                         "39000 matches the existing 100%% cell")
    ap.add_argument("--eval-every-steps", type=int, default=390,
                    help="evaluate on a fixed STEP cadence so every prune "
                         "fraction gets the same number of evaluations")
    ap.add_argument("--sweep", nargs="+", default=None)
    args = ap.parse_args()

    if args.sweep:
        analyze_sweep(args.sweep)
        return
    if args.self_test:
        self_test()
        return
    if args.analyze:
        analyze(args.analyze)
        return
    if args.seed is None or args.output_path is None or not args.static_scores:
        ap.error("--seed, --output-path and --static-scores are required")
    run(args)


if __name__ == "__main__":
    main()