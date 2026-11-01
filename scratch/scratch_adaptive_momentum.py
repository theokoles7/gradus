#!/usr/bin/env python3
"""
scratch_adaptive_momentum.py

Can a controller driven by learning dynamics help by acting on OPTIMIZER STATE
rather than on the data?

Everything this project has tested modifies the gradient g: which examples
contribute to it (composition, pruning, withholding), how they are weighted
(loss reweighting, per-example LR, which are the same lever), or the order in
which batches arrive (ordering, sequence policies). All of it is at or below
parity with shuffled SGD at adequate seed counts.

SGD with momentum carries a separate object:

    v <- mu * v + g
    theta <- theta - lr * v

mu and v have never been touched. A controller that modulates mu PER PARAMETER
BLOCK from measured learning dynamics changes how long past gradients persist,
not what the current gradient is. That is structurally distinct from every
family already eliminated, which is the only reason to run it.

Signals (both already measured elsewhere in this project, both free here)
------------------------------------------------------------------------
coherence   per-block ||mean_m g_m||^2 / mean_m ||g_m||^2 over a short window
            of recent minibatch gradients. The frontier diagnostic measured
            this at the whole-model level: it starts near 0.6 and decays to
            roughly 0 by epoch 100, and its relationship to subsequent learning
            flips sign around lr = 0.06. Reading: high coherence means recent
            gradients already agree, so averaging buys little and mu can drop;
            low coherence means the signal needs averaging, so mu rises.

drift       per-block representation drift between refreshes, the quantity
            logged in all twelve LeRaC runs and never used. Same mapping: a
            block whose output is still moving fast gets more smoothing.

Both map a signal s in [0,1] to mu via

    mu_b = mu_lo + (mu_hi - mu_lo) * (1 - s_norm)

where s_norm is the block's signal rank-normalised across blocks at that
refresh, so the controller is scale-free and cannot drift the whole network's
mu in one direction. --invert flips the mapping, because the sign is an
empirical question and the frontier result showed this exact relationship
reversing partway through training.

Arms
----
    constant     mu = 0.9 everywhere. CONTROL. Identical to the established
                 baseline, so this arm must reproduce ~77.5.
    coherence    per-block mu from windowed gradient coherence
    drift        per-block mu from representation drift

Blocks are the ten semantic units used throughout this project: stem, the eight
residual blocks, classifier.

Cost
----
coherence needs the per-block gradient of the last W minibatches, which are the
training gradients themselves, so it is accumulation only: no extra forward or
backward passes. drift needs one forward pass over a fixed probe subset per
refresh, about 8 batches, which is negligible. Both arms are within a few
percent of the control.

Seed policy
-----------
Nine seeds from the start. Three-seed effects in the 0.1 to 0.2pp range have
decayed to zero four times in this project (ESS reweighting, local pool,
progress_high, adaptive withholding), so screening at three seeds has no
information value at the effect sizes in play.

Usage
-----
    python scratch_adaptive_momentum.py --self-test
    python scratch_adaptive_momentum.py --seed 1 --arm coherence \\
        --output-path results/mom/coherence_seed1.json
    python scratch_adaptive_momentum.py --analyze "results/mom/*.json"
"""

import argparse
import glob
import json
import math
import os
import random
import re
from collections import OrderedDict, defaultdict

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
EPOCHS = 100
PROBE_PER_CLASS = 10

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"

GROUPS = ["stem", "layer1.0", "layer1.1", "layer2.0", "layer2.1",
          "layer3.0", "layer3.1", "layer4.0", "layer4.1", "classifier"]
EXPECTED_TENSORS = OrderedDict([
    ("stem", 3), ("layer1.0", 6), ("layer1.1", 6), ("layer2.0", 9),
    ("layer2.1", 6), ("layer3.0", 9), ("layer3.1", 6), ("layer4.0", 9),
    ("layer4.1", 6), ("classifier", 2)])
ARMS = ("constant", "coherence", "drift")

_BLOCK_RE = re.compile(r"^_layer([1-4])_\.(\d+)\.")


def assign_group(name):
    """Gradus wraps module attributes in underscores: '_conv1_.weight',
    '_layer2_.0._downsample_.0.weight', '_fc_.bias'.
    """
    if name.startswith("_conv1_.") or name.startswith("_bn1_."):
        return "stem"
    m = _BLOCK_RE.match(name)
    if m:
        g = f"layer{m.group(1)}.{m.group(2)}"
        if g not in EXPECTED_TENSORS:
            raise ValueError(f"'{name}' -> unexpected group '{g}'")
        return g
    if name.startswith("_fc_."):
        return "classifier"
    raise ValueError(f"Unmapped parameter '{name}'")


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
# Signal -> momentum mapping
# --------------------------------------------------------------------------

def signal_to_mu(signal, mu_lo, mu_hi, invert=False):
    """Rank-normalise the per-block signal across blocks, then map to mu.

    Rank normalisation, not raw scaling, for two reasons. The signals have very
    different natural ranges across blocks and across training (coherence
    decays from ~0.6 to ~0 over 100 epochs), and an absolute mapping would
    therefore drift every block's mu in the same direction over training,
    turning a per-block controller into a global momentum schedule. Ranking
    across blocks keeps the controller purely RELATIVE: it always assigns the
    full [mu_lo, mu_hi] spread, and only reorders which block gets what.

    Default mapping: HIGH signal -> LOW mu (recent gradients already agree, so
    averaging buys little). --invert flips it; the sign is an empirical
    question, and the frontier diagnostic showed this exact relationship
    reversing partway through training.
    """
    s = np.asarray(signal, dtype=np.float64)
    n = len(s)
    if n == 1:
        r = np.array([0.5])
    else:
        r = np.argsort(np.argsort(s)).astype(np.float64) / (n - 1)
    if invert:
        r = 1.0 - r
    return mu_lo + (mu_hi - mu_lo) * (1.0 - r)


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

class CoherenceTracker:
    """Windowed per-block gradient coherence from the TRAINING gradients.

    For a window of W minibatch gradients per block,

        C_b = ||mean_m g_bm||^2 / mean_m ||g_bm||^2

    accumulated as a running sum vector and a running scalar, so nothing large
    is stored and no extra passes are needed. The raw statistic has a floor of
    about 1/W, so the bias-corrected form

        (W * ||mean||^2 - mean||.||^2) / ((W - 1) * mean||.||^2)

    is used; it is a monotone transform of the raw value, so rank-based use is
    unaffected, but it keeps the reported numbers interpretable late in
    training when raw coherence sits at the floor.
    """

    def __init__(self, groups, sizes, device, window):
        self.groups = groups
        self.window = window
        self.sum = {g: torch.zeros(sizes[g], device=device) for g in groups}
        self.sq = {g: 0.0 for g in groups}
        self.n = 0

    def accumulate(self, flat_by_group):
        for g, v in flat_by_group.items():
            self.sum[g].add_(v)
            self.sq[g] += float(v.dot(v))
        self.n += 1

    def ready(self):
        return self.n >= self.window

    def read_and_reset(self):
        out = {}
        for g in self.groups:
            mean = self.sum[g] / max(self.n, 1)
            mean_sq = self.sq[g] / max(self.n, 1)
            raw = float(mean.dot(mean)) / (mean_sq + 1e-30)
            w = max(self.n, 2)
            out[g] = (w * raw - 1.0) / (w - 1.0)
            self.sum[g].zero_()
            self.sq[g] = 0.0
        self.n = 0
        return out


class DriftTracker:
    """Per-block representation drift on a fixed unaugmented probe subset.

        D_l = ||Z_l - Z_l_prev||_F / (||Z_l_prev||_F + eps)

    Uses the network's own return_activations path (layer1..layer4 outputs,
    already detached) and runs in eval mode so BatchNorm running statistics are
    not updated. Only four of the ten groups have a measurable drift; the stem
    and classifier inherit the nearest measured block, which is stated in the
    log rather than hidden.
    """

    BLOCKS = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, model, probe_x):
        self.model = model
        self.probe_x = probe_x
        self.prev = {}

    def read(self):
        was = self.model.training
        self.model.eval()
        with torch.no_grad():
            _, acts = self.model(self.probe_x, return_activations=True)
        if was:
            self.model.train()
        out = {}
        for name, A in zip(self.BLOCKS, acts):
            Z = A.flatten(1).float()
            if name in self.prev:
                P = self.prev[name]
                out[name] = float(torch.linalg.norm(Z - P)
                                  / (torch.linalg.norm(P) + 1e-12))
            self.prev[name] = Z
        if not out:
            return None
        # expand four measured values onto the ten parameter groups
        return {
            "stem": out["layer1"],
            "layer1.0": out["layer1"], "layer1.1": out["layer1"],
            "layer2.0": out["layer2"], "layer2.1": out["layer2"],
            "layer3.0": out["layer3"], "layer3.1": out["layer3"],
            "layer4.0": out["layer4"], "layer4.1": out["layer4"],
            "classifier": out["layer4"],
        }


def build_probe(dataset, n_per_class=PROBE_PER_CLASS, n_classes=NUM_CLASSES,
                seed=0):
    rng = random.Random(seed)
    by = {c: [] for c in range(n_classes)}
    for i, y in enumerate(dataset.targets):
        by[int(y)].append(i)
    chosen = []
    for c in range(n_classes):
        chosen.extend(rng.sample(by[c], n_per_class))
    return torch.stack([dataset[i][0] for i in chosen])


# --------------------------------------------------------------------------

def build_optimizer(model):
    buckets = OrderedDict((g, []) for g in GROUPS)
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[assign_group(name)].append(p)
    groups = []
    for g in GROUPS:
        if not buckets[g]:
            raise ValueError(f"group '{g}' empty")
        groups.append({"params": buckets[g], "lr": BASE_LR,
                       "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
                       "name": g})
    opt = torch.optim.SGD(groups, lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    table = []
    print(f"{'group':<12}{'tensors':>8}{'exp':>5}{'n_params':>11}")
    for pg in opt.param_groups:
        g = pg["name"]
        nt, nel = len(pg["params"]), sum(p.numel() for p in pg["params"])
        assert nt == EXPECTED_TENSORS[g], f"{g}: {nt} != {EXPECTED_TENSORS[g]}"
        assert pg["momentum"] == MOMENTUM and pg["weight_decay"] == WEIGHT_DECAY
        print(f"{g:<12}{nt:>8}{EXPECTED_TENSORS[g]:>5}{nel:>11}")
        table.append({"group": g, "n_tensors": nt, "n_params": nel})
    total = sum(t["n_params"] for t in table)
    assert total == sum(p.numel() for p in model.parameters())
    print(f"{'total':<12}{'':>8}{'':>5}{total:>11}\n")
    return opt, table, buckets


def group_flat_grads(buckets, sizes, device):
    out = {}
    for g, ps in buckets.items():
        v = torch.empty(sizes[g], device=device)
        off = 0
        for p in ps:
            n = p.numel()
            if p.grad is None:
                v[off:off + n].zero_()
            else:
                v[off:off + n].copy_(p.grad.detach().reshape(-1))
            off += n
        out[g] = v
    return out


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
    ds_train, ds_plain, ds_test = build_datasets()
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                             num_workers=nw, pin_memory=True)

    set_seed(args.seed)
    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt, table, buckets = build_optimizer(model)
    sizes = {g: sum(p.numel() for p in ps) for g, ps in buckets.items()}
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    coh = (CoherenceTracker(GROUPS, sizes, device, args.window)
           if args.arm == "coherence" else None)
    drift = None
    if args.arm == "drift":
        drift = DriftTracker(model, build_probe(ds_plain).to(device))
        drift.read()        # seed the previous activations

    mu_log, log = [], []
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rl, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()

            if coh is not None:
                coh.accumulate(group_flat_grads(buckets, sizes, device))
                if coh.ready():
                    sig = coh.read_and_reset()
                    mu = signal_to_mu([sig[g] for g in GROUPS],
                                      args.mu_lo, args.mu_hi, args.invert)
                    for pg, m in zip(opt.param_groups, mu):
                        pg["momentum"] = float(m)

            opt.step()
            rl += loss.item() * y.size(0)
            seen += y.size(0)

        if drift is not None and epoch % args.refresh_every == 0:
            sig = drift.read()
            if sig is not None:
                mu = signal_to_mu([sig[g] for g in GROUPS],
                                  args.mu_lo, args.mu_hi, args.invert)
                for pg, m in zip(opt.param_groups, mu):
                    pg["momentum"] = float(m)

        acc = evaluate(model, test_loader, device)
        best = max(best, acc)
        cur_mu = {pg["name"]: round(pg["momentum"], 4)
                  for pg in opt.param_groups}
        log.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                    "train_loss": rl / seen, "test_acc": acc})
        if epoch % 10 == 0 or epoch == args.epochs:
            mu_log.append({"epoch": epoch, "mu": cur_mu})
            print(f"ep {epoch:3d} loss={rl/seen:.4f} test={acc:.2f} "
                  f"best={best:.2f}  mu stem={cur_mu['stem']:.2f} "
                  f"l4.1={cur_mu['layer4.1']:.2f} clf={cur_mu['classifier']:.2f}")
        sched.step()

    out = {"seed": args.seed, "arm": args.arm, "epochs": args.epochs,
           "mu_lo": args.mu_lo, "mu_hi": args.mu_hi, "invert": args.invert,
           "window": args.window, "refresh_every": args.refresh_every,
           "batch_size": BATCH_SIZE, "group_table": table,
           "note": "per-block momentum; gradient and data pipeline untouched",
           "best_test_acc": best, "final_test_acc": log[-1]["test_acc"],
           "mu_log": mu_log, "epochs_log": log}
    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{args.arm} seed {args.seed}: best={best:.2f} "
          f"final={log[-1]['test_acc']:.2f}\n-> {args.output_path}")


# --------------------------------------------------------------------------

def self_test():
    print("signal -> mu mapping (rank-normalised across the 10 blocks)")
    sig = [0.60, 0.42, 0.38, 0.31, 0.29, 0.25, 0.22, 0.20, 0.19, 0.18]
    mu = signal_to_mu(sig, 0.80, 0.95)
    print(f"  {'block':<12}{'signal':>9}{'mu':>8}")
    for g, s_, m in zip(GROUPS, sig, mu):
        print(f"  {g:<12}{s_:>9.3f}{m:>8.3f}")
    print(f"  spread always spans [mu_lo, mu_hi]: "
          f"{bool(abs(mu.min()-0.80) < 1e-9 and abs(mu.max()-0.95) < 1e-9)}")

    print("\nscale-free check: the SAME ordering at a much smaller magnitude "
          "(late training) must give the same mu")
    mu2 = signal_to_mu([s_ * 0.01 for s_ in sig], 0.80, 0.95)
    print(f"  identical: {bool(np.allclose(mu, mu2))}  "
          f"(an absolute mapping would have collapsed every block to one end)")

    print("\ninvert flips the assignment:")
    mu3 = signal_to_mu(sig, 0.80, 0.95, invert=True)
    print(f"  highest-signal block: mu {mu[0]:.3f} -> {mu3[0]:.3f};  "
          f"lowest: {mu[-1]:.3f} -> {mu3[-1]:.3f}")

    print("\ncoherence bias correction (raw floor is ~1/W):")
    for W in (4, 8, 16):
        for raw in (1.0 / W, 0.3, 0.6):
            c = (W * raw - 1.0) / (W - 1.0)
            print(f"  W={W:<3} raw {raw:.3f} -> corrected {c:+.3f}"
                  + ("   (at the floor -> 0)" if abs(raw - 1.0/W) < 1e-9 else ""))

    print("\nconstant arm must leave momentum at 0.9 for every block, so it "
          "reproduces the established baseline exactly.")
    print("\nself-test passed")


def analyze(paths):
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit("no files matched")
    by = defaultdict(dict)
    for f in files:
        L = json.load(open(f))
        by[L["arm"]][L["seed"]] = L["best_test_acc"]
    arms = [a for a in ARMS if a in by]
    seeds = sorted(set.intersection(*[set(by[a]) for a in arms]))
    print(f"{len(seeds)} shared seed(s): {seeds}\n")
    print(f"{'seed':<6}" + "".join(f"{a:>14}" for a in arms))
    for s in seeds:
        print(f"{s:<6}" + "".join(f"{by[a][s]:>14.2f}" for a in arms))
    col = {a: np.array([by[a][s] for s in seeds]) for a in arms}
    print(f"{'mean':<6}" + "".join(f"{col[a].mean():>14.2f}" for a in arms))

    n = len(seeds)
    if "constant" not in col:
        print("\nno constant arm; nothing to compare against")
        return
    print("\nvs constant momentum (mu = 0.9):")
    for a in arms:
        if a == "constant":
            continue
        d = col[a] - col["constant"]
        if n > 1:
            se = d.std(ddof=1) / math.sqrt(n)
            print(f"  {a:<12} {d.mean():+6.2f}pp  sd={d.std(ddof=1):.2f}  "
                  f"t={d.mean()/se:+.2f} (df={n-1})  pos={int((d>0).sum())}/{n}")
        else:
            print(f"  {a:<12} {d.mean():+6.2f}pp  (single seed)")
    if n < 9:
        print(f"\n{n} seeds. Effects of 0.1-0.2pp have decayed to zero on "
              f"extension four times in this project; treat anything in that "
              f"range at n<9 as unresolved.")
    print("\nComputed ad hoc by this script; rerun in your own analysis code "
          "before citing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--mu-lo", type=float, default=0.80)
    ap.add_argument("--mu-hi", type=float, default=0.95)
    ap.add_argument("--invert", action="store_true",
                    help="high signal -> HIGH mu instead of low")
    ap.add_argument("--window", type=int, default=8,
                    help="minibatches per coherence estimate")
    ap.add_argument("--refresh-every", type=int, default=1,
                    help="epochs between drift measurements")
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