#!/usr/bin/env python3
"""
scratch_lerac_cosine.py

LeRaC-Artifact-Cosine: model-level LR curriculum applied as a multiplicative
per-group factor on an unchanged global cosine schedule.

Frozen per docs/prereg_lerac_cosine.md. Do not change the schedule, group table,
or gate without a dated amendment to that document.

VERIFICATION (must pass before seed 1 of the real experiment)
-------------------------------------------------------------
The load-bearing check is against the EXISTING shuffled baseline script, not
against this script's own control arm. Running --arm lerac --force-unit-multipliers
and --arm control here executes the identical code path, so comparing those two
proves only that the flag plumbing works; it cannot catch a grouping error, a
weight-decay inheritance error, or a scheduler indexing error.

    # 1. code-path smoke test (cheap, proves little)
    python scratch_lerac_cosine.py --arm lerac --force-unit-multipliers \
        --seed 1 --output-path results/lerac/verify_unit_seed1.json
    python scratch_lerac_cosine.py --arm control \
        --seed 1 --output-path results/lerac/verify_control_seed1.json
    python scratch_lerac_cosine.py --compare \
        results/lerac/verify_unit_seed1.json results/lerac/verify_control_seed1.json

    # 2. THE REAL CHECK: forced-unit run vs the existing baseline, same seed
    python scratch_lerac_cosine.py --compare \
        results/lerac/verify_unit_seed1.json <baseline_seed1.json>

For step 2 the baseline JSON must expose a per-epoch accuracy series. If it does
not, add that logging to the baseline script first.

EXPERIMENT (only after step 2 reports IDENTICAL)
------------------------------------------------
    for s in 1 2 3 4 5 6; do
      python scratch_lerac_cosine.py --arm control --seed $s --drift-probe \
          --output-path results/lerac/control_seed$s.json
      python scratch_lerac_cosine.py --arm lerac   --seed $s --drift-probe \
          --output-path results/lerac/lerac_seed$s.json
    done

METRIC NAMING
-------------
Gradus' Dataset.test_loader is built from the CIFAR-100 *test* split, so what
this script records as "test_acc" is test accuracy, and "best_test_acc" is the
maximum over all epochs on the test set. Both arms are treated identically so
the paired delta is unaffected, but the writeup must call it test accuracy and
state that the max is taken over epochs.
"""

import argparse
import json
import os
import random
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

# RECONCILE: confirm these import paths against the installed package layout.
from gradus.networks.resnet.resnet_18 import ResNet18
from gradus.datasets.cifar_100 import CIFAR_100
from gradus.utilities import get_system_core_count


# --------------------------------------------------------------------------
# Frozen specification (prereg section 3)
# --------------------------------------------------------------------------

BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
K_RELEASE = 7          # completed epochs after which all groups are released
SCALE_PER_EPOCH = 10.0

GROUP_ORDER = [
    "stem",
    "layer1.0", "layer1.1",
    "layer2.0", "layer2.1",
    "layer3.0", "layer3.1",
    "layer4.0", "layer4.1",
    "classifier",
]

INITIAL_LR = OrderedDict([
    ("stem",       1e-1),
    ("layer1.0",   1e-2),
    ("layer1.1",   1e-3),
    ("layer2.0",   1e-4),
    ("layer2.1",   1e-5),
    ("layer3.0",   1e-6),
    ("layer3.1",   1e-7),
    ("layer4.0",   1e-8),
    ("layer4.1",   1e-8),
    ("classifier", 1e-8),
])

# Expected tensor counts per group for Gradus ResNet-18 [2,2,2,2].
# Blocks carrying a downsample branch have 3 extra tensors.
EXPECTED_TENSORS = OrderedDict([
    ("stem",       3),
    ("layer1.0",   6), ("layer1.1", 6),
    ("layer2.0",   9), ("layer2.1", 6),
    ("layer3.0",   9), ("layer3.1", 6),
    ("layer4.0",   9), ("layer4.1", 6),
    ("classifier", 2),
])

# r_j = eta_j(0) / BASE_LR
R = OrderedDict((k, v / BASE_LR) for k, v in INITIAL_LR.items())


def multiplier(group_name, e, force_unit=False):
    """a_j(e) = min(1, 10^e * r_j).

    e is the number of COMPLETED training epochs before the current one.
    Training epoch 1 has e=0; every group is fully released from e=K_RELEASE,
    so the first full epoch at the global LR is training epoch 8.
    """
    if force_unit:
        return 1.0
    if e >= K_RELEASE:
        return 1.0
    return min(1.0, (SCALE_PER_EPOCH ** e) * R[group_name])


# --------------------------------------------------------------------------
# Data (Gradus CIFAR-100, shuffled-baseline path)
# --------------------------------------------------------------------------

# Must match gradus.datasets.cifar_100 exactly.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"


def get_cifar100_loaders(seed, batch_size, epochs):
    """Shuffled-baseline data path, plus an unaugmented training view.

    metric=None routes Dataset.train_loader to a plain shuffled DataLoader with
    no curriculum sampler, which is the baseline condition. shuffle=True is
    passed explicitly because the Gradus default is False.

    train_eval_set is a separate CIFAR100 instance over the same training
    indices with the test-time transform. It is used only by the drift probe
    and never by the optimizer.
    """
    ds = CIFAR_100(
        epochs=epochs,
        root=DATA_ROOT,
        batch_size=batch_size,
        shuffle=True,
        metric=None,
        schedule=None,
        normalize_classes=False,
        seed=seed,
    )

    train_eval_set = CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]),
    )

    return ds.train_loader, ds.test_loader, train_eval_set, ds


# --------------------------------------------------------------------------
# Parameter grouping
# --------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"^_layer([1-4])_\.(\d+)\.")


def assign_group(param_name):
    """Map a Gradus ResNet-18 named parameter to one of the ten curriculum units.

    Gradus wraps module attributes in underscores, so parameter names look like
    '_conv1_.weight', '_layer2_.0._downsample_.0.weight', '_fc_.bias'.
    Block-contained BN, bias, and downsample parameters inherit the block rate.
    """
    if param_name.startswith("_conv1_.") or param_name.startswith("_bn1_."):
        return "stem"

    m = _BLOCK_RE.match(param_name)
    if m:
        group = f"layer{m.group(1)}.{m.group(2)}"
        if group not in INITIAL_LR:
            raise ValueError(
                f"'{param_name}' maps to '{group}', which is not in the frozen "
                f"group table. Expected [2,2,2,2] blocks."
            )
        return group

    if param_name.startswith("_fc_."):
        return "classifier"

    raise ValueError(
        f"Unmapped parameter '{param_name}'. Every parameter must land in a "
        f"declared group; fix assign_group rather than adding a fallback."
    )


def build_optimizer(model):
    buckets = OrderedDict((g, []) for g in GROUP_ORDER)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        buckets[assign_group(name)].append((name, p))

    groups = []
    for g in GROUP_ORDER:
        params = [p for _, p in buckets[g]]
        if len(params) == 0:
            raise ValueError(f"Group '{g}' is empty; grouping logic is wrong.")
        groups.append({
            "params": params,
            "lr": INITIAL_LR[g],
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "name": g,
        })

    opt = torch.optim.SGD(groups, lr=BASE_LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    table = assert_group_table(opt)
    assert_full_coverage(model, opt)
    return opt, table


def assert_group_table(opt):
    """Prereg section 5: hard assertions at construction, plus a logged table."""
    assert len(opt.param_groups) == 10, \
        f"expected 10 groups, got {len(opt.param_groups)}"

    table = []
    print(f"{'group':<12} {'n_params':>11} {'tensors':>8} {'exp':>4} "
          f"{'lr':>10} {'mom':>5} {'wd':>8}")
    for pg in opt.param_groups:
        g = pg["name"]
        n_el = sum(p.numel() for p in pg["params"])
        n_t = len(pg["params"])
        assert abs(pg["lr"] - INITIAL_LR[g]) < 1e-20, \
            f"{g}: lr {pg['lr']} != frozen {INITIAL_LR[g]}"
        assert pg["momentum"] == MOMENTUM, f"{g}: momentum {pg['momentum']}"
        assert pg["weight_decay"] == WEIGHT_DECAY, f"{g}: wd {pg['weight_decay']}"
        flag = "" if n_t == EXPECTED_TENSORS[g] else "  <-- UNEXPECTED"
        print(f"{g:<12} {n_el:>11} {n_t:>8} {EXPECTED_TENSORS[g]:>4} "
              f"{pg['lr']:>10.2e} {pg['momentum']:>5} "
              f"{pg['weight_decay']:>8.0e}{flag}")
        table.append({"group": g, "n_params": n_el, "n_tensors": n_t,
                      "expected_tensors": EXPECTED_TENSORS[g],
                      "initial_lr": pg["lr"]})

    mismatched = [t["group"] for t in table
                  if t["n_tensors"] != t["expected_tensors"]]
    if mismatched:
        raise AssertionError(
            f"tensor counts differ from expectation in {mismatched}. "
            f"The grouping is wrong; do not run."
        )
    print()
    return table


def assert_full_coverage(model, opt):
    """Every trainable parameter appears in exactly one group."""
    in_groups = sum(len(pg["params"]) for pg in opt.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert in_groups == trainable, \
        f"{in_groups} params in groups, {trainable} trainable in model"
    seen = set()
    for pg in opt.param_groups:
        for p in pg["params"]:
            assert id(p) not in seen, "parameter assigned to two groups"
            seen.add(id(p))


# --------------------------------------------------------------------------
# Global cosine schedule, read from a throwaway driver
# --------------------------------------------------------------------------

class GlobalCosine:
    """Drives CosineAnnealingLR on a dummy parameter so eta_g(e) is read from a
    real scheduler rather than reconstructed analytically, while the real
    optimizer's group LRs stay writable without corrupting scheduler state.
    """

    def __init__(self, epochs=EPOCHS, base_lr=BASE_LR):
        self._p = torch.nn.Parameter(torch.zeros(1))
        self._opt = torch.optim.SGD([self._p], lr=base_lr)
        self._sched = CosineAnnealingLR(self._opt, T_max=epochs)

    def current(self):
        return self._opt.param_groups[0]["lr"]

    def step(self):
        # No-op: the dummy parameter has no grad, so SGD updates nothing. This
        # only increments the optimizer's step count, which silences PyTorch's
        # "step() called before optimizer.step()" warning. LR values unchanged.
        self._opt.step()
        self._sched.step()


def apply_lrs(opt, eta_g, e, force_unit=False):
    """eta_j(e) = eta_g(e) * a_j(e). Assigned, never multiplied in place."""
    applied = {}
    for pg in opt.param_groups:
        pg["lr"] = eta_g * multiplier(pg["name"], e, force_unit=force_unit)
        applied[pg["name"]] = pg["lr"]
    return applied


# --------------------------------------------------------------------------
# Drift probe (prereg section 6): logged only, never affects training
# --------------------------------------------------------------------------

class DriftProbe:
    """Uses the network's own return_activations path, which yields the
    layer1..layer4 outputs already detached.
    """

    BLOCKS = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, model, probe_x):
        self.model = model
        self.probe_x = probe_x
        self.prev = {}

    @torch.no_grad()
    def measure(self):
        was_training = self.model.training
        self.model.eval()                      # do not perturb BN running stats
        _, activations = self.model(self.probe_x, return_activations=True)
        if was_training:
            self.model.train()

        out = {}
        for name, A in zip(self.BLOCKS, activations):
            Z = A.flatten(1).float().cpu()
            if name in self.prev:
                P = self.prev[name]
                rel = (torch.linalg.norm(Z - P) /
                       (torch.linalg.norm(P) + 1e-12)).item()
                cos = F.cosine_similarity(Z, P, dim=1).mean().item()
                out[name] = {"rel_frobenius_drift": rel, "mean_cosine": cos}
            self.prev[name] = Z
        return out


def build_probe_batch(dataset, n_per_class=10, num_classes=100, seed=0):
    """Fixed stratified probe over the unaugmented training view."""
    rng = random.Random(seed)
    targets = getattr(dataset, "targets", None)
    by_class = {c: [] for c in range(num_classes)}
    if targets is not None:
        for idx, y in enumerate(targets):
            by_class[int(y)].append(idx)
    else:
        for idx in range(len(dataset)):
            by_class[int(dataset[idx][1])].append(idx)
    chosen = []
    for c in range(num_classes):
        chosen.extend(rng.sample(by_class[c], n_per_class))
    return torch.stack([dataset[i][0] for i in chosen])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Required for the exact-equality verification comparison.
    # RECONCILE: the baseline script must use the same flags.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def run(args):
    if os.path.exists(args.output_path):
        raise SystemExit(
            f"{args.output_path} exists. Use a distinctive --output-path; "
            f"silent overwrites have cost this project results before."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_loader, test_loader, train_eval_dataset, ds = get_cifar100_loaders(
        seed=args.seed, batch_size=args.batch_size, epochs=args.epochs
    )

    model = ResNet18(
        input_shape=ds.input_shape,
        num_classes=ds.num_classes,
        zero_init_residual=False,
    ).to(device)

    opt, group_table = build_optimizer(model)
    cosine = GlobalCosine(epochs=args.epochs, base_lr=BASE_LR)
    criterion = nn.CrossEntropyLoss()

    probe = None
    if args.drift_probe:
        probe_x = build_probe_batch(
            train_eval_dataset, num_classes=ds.num_classes
        ).to(device)
        probe = DriftProbe(model, probe_x)

    force_unit = args.force_unit_multipliers or (args.arm == "control")

    log = {
        "arm": args.arm,
        "seed": args.seed,
        "force_unit_multipliers": bool(args.force_unit_multipliers),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": get_system_core_count(),
        "small_stem": bool(model._small_stem_),
        "dataset": ds.dict,
        "input_shape": list(ds.input_shape),
        "initial_lr_table": dict(INITIAL_LR),
        "k_release": K_RELEASE,
        "group_table": group_table,
        "metric_note": "test_acc is CIFAR-100 test split; best is max over epochs",
        "epochs_log": [],
    }

    best_test = 0.0
    for epoch in range(1, args.epochs + 1):
        e = epoch - 1                      # completed epochs before this one
        eta_g = cosine.current()
        applied = apply_lrs(opt, eta_g, e, force_unit=force_unit)

        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)

        test_acc = evaluate(model, test_loader, device)
        best_test = max(best_test, test_acc)
        drift = probe.measure() if probe is not None else None

        log["epochs_log"].append({
            "epoch": epoch,
            "e": e,
            "eta_g": eta_g,
            "group_lrs": applied,
            "train_loss": run_loss / seen,
            "test_acc": test_acc,
            "drift": drift,
        })
        print(f"ep {epoch:3d}  e={e:3d}  eta_g={eta_g:.6f}  "
              f"loss={run_loss / seen:.4f}  test={test_acc:.2f}  "
              f"best={best_test:.2f}")

        cosine.step()                      # advance after the epoch, per prereg

    log["best_test_acc"] = best_test
    log["final_test_acc"] = log["epochs_log"][-1]["test_acc"]

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nbest={best_test:.2f}  final={log['final_test_acc']:.2f}  "
          f"-> {args.output_path}")


# --------------------------------------------------------------------------
# Verification comparison (prereg section 5)
# --------------------------------------------------------------------------

def _acc_series(blob):
    """Per-epoch accuracy series from this script's log or a baseline log.

    RECONCILE: add the baseline's key to BASELINE_KEYS if it differs.
    """
    BASELINE_KEYS = ("test_acc_per_epoch", "val_acc_per_epoch",
                     "accuracy_per_epoch")

    if isinstance(blob, list):
        return list(blob)
    if isinstance(blob, dict):
        if "epochs_log" in blob:
            first = blob["epochs_log"][0]
            key = "test_acc" if "test_acc" in first else "val_acc"
            return [ep[key] for ep in blob["epochs_log"]]
        for k in BASELINE_KEYS:
            if k in blob:
                return list(blob[k])
    raise SystemExit(
        "Could not find a per-epoch accuracy series. Add one to the baseline "
        "log, or add its key to BASELINE_KEYS in _acc_series."
    )


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    sa, sb = _acc_series(a), _acc_series(b)

    if len(sa) != len(sb):
        print(f"length differs: {len(sa)} vs {len(sb)}")
        print("MISMATCH: do not proceed to seed 1")
        return False

    ok = True
    for i, (va, vb) in enumerate(zip(sa, sb), start=1):
        if va != vb:
            print(f"ep {i}: accuracy differs  {va} vs {vb}")
            ok = False

    # eta_g series, when both files carry it
    if all(isinstance(x, dict) and "epochs_log" in x for x in (a, b)):
        for ea, eb in zip(a["epochs_log"], b["epochs_log"]):
            if abs(ea["eta_g"] - eb["eta_g"]) > 1e-12:
                print(f"ep {ea['epoch']}: eta_g differs "
                      f"{ea['eta_g']} vs {eb['eta_g']}")
                ok = False

    print("IDENTICAL" if ok else "MISMATCH: do not proceed to seed 1")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["lerac", "control"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=128)  # RECONCILE
    ap.add_argument("--output-path", type=str)
    ap.add_argument("--force-unit-multipliers", action="store_true",
                    help="verification only: a_j(e)=1 for all j,e")
    ap.add_argument("--drift-probe", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if args.arm is None or args.seed is None or args.output_path is None:
        ap.error("--arm, --seed and --output-path are required")
    run(args)


if __name__ == "__main__":
    main()