#!/usr/bin/env python3
"""
scratch_frontier_diagnostic.py

MEASUREMENT ONLY. Plain shuffled training, no sampling change, no intervention.

Question
--------
Does shuffled ResNet-18/CIFAR-100 exhibit a measurable, temporally moving
gradient-activity frontier across difficulty bands? And does gradient coherence
predict subsequent learning, or predict its absence?

The second question is not decoration. Coherence

    C_k = ||mean_m g_km||^2 / mean_m ||g_km||^2

is gradient diversity inverted. This project's confirmed finding is that
destroying gradient diversity is what sank every composition-based mechanism
(the ratchet's -0.40pp traced to a measured diversity ratio as low as 0.14 in
epochs 1-5; shuffled beats sequential by +0.5 to +1.8pp across eight
architecture/dataset configurations). A frontier curriculum that steers toward
high-coherence bands is, mechanically, a procedure for lowering diversity. So
the SIGN of the coherence term is an open empirical question here, not a design
assumption, and this script is built to answer it.

Two partitions, same run
------------------------
  static  : bands frozen from a fixed per-sample score. Either external scores
            (--static-scores) or, by default, the epoch-1 unaugmented per-sample
            loss, frozen thereafter. This is the "static offline score" analogue
            whose progressive divergence from the model's real ordering was the
            diagnosed failure of the original ratchet.
  dynamic : bands recomputed every epoch from the model's own current state
            (EMA of per-sample loss, or margin).

If the dynamic partition shows a moving ridge and the static one does not, that
is direct evidence that score staleness, not curriculum per se, has been the
problem.

Pre-registered criteria (evaluated by --analyze; declare before looking)
-----------------------------------------------------------------------
  R1 RIDGE      argmax_k A_k(t) advances by >= 2 bands between epoch 10 and 70,
                with a non-negative Spearman trend in argmax vs epoch.
  R2 SEPARATION at some epoch, the spread of A_k across bands exceeds the
                within-band epoch-to-epoch variation (ratio > 2).
  R3 SIGN       Spearman(C_k(t), dL_k(t -> t+DELTA)) over all (k, t).
                POSITIVE  -> coherence predicts learning; GAFC's term is right.
                NEGATIVE  -> coherence predicts LESS learning; GAFC steers the
                             wrong way and the mechanism should not be built.

R1 and R2 failing means no exploitable frontier; stop. R3 negative means the
frontier may exist but coherence is the wrong readiness signal.

Cost
----
Per epoch: 2 partitions x K bands x M minibatches backward passes (default
2x5x8 = 80), plus one forward-only scoring pass over the training set. Roughly
25-30% over a plain run. Run 2-3 seeds; one run cannot separate structure from
run-to-run variation.

Usage
-----
    python scratch_frontier_diagnostic.py --seed 1 \
        --output-path results/frontier/frontier_seed1.json
    python scratch_frontier_diagnostic.py --analyze \
        results/frontier/frontier_seed1.json
"""

import argparse
import copy
import json
import math
import os
import random
from collections import OrderedDict

import numpy as np

# torch, torchvision and gradus are imported lazily inside _lazy_imports() so
# that --analyze works in an environment without them (laptop, CI, notebook).
torch = None
nn = None
DataLoader = None
CosineAnnealingLR = None
CIFAR100 = None
Compose = Normalize = RandomCrop = RandomHorizontalFlip = ToTensor = None
ResNet18 = None
get_system_core_count = None


def _lazy_imports():
    global torch, nn, DataLoader, CosineAnnealingLR, CIFAR100
    global Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
    global ResNet18, get_system_core_count

    import torch as _torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader as _DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR as _Cos
    from torchvision.datasets import CIFAR100 as _C100
    from torchvision.transforms import (Compose as _Cp, Normalize as _Nm,
                                        RandomCrop as _RC,
                                        RandomHorizontalFlip as _RHF,
                                        ToTensor as _TT)
    # RECONCILE: confirm against the installed package layout.
    from gradus.networks.resnet.resnet_18 import ResNet18 as _R18
    from gradus.utilities import get_system_core_count as _cores

    torch, nn, DataLoader, CosineAnnealingLR, CIFAR100 = (
        _torch, _nn, _DataLoader, _Cos, _C100)
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor = (
        _Cp, _Nm, _RC, _RHF, _TT)
    ResNet18, get_system_core_count = _R18, _cores


# --------------------------------------------------------------------------
# Regime (unchanged from the established setup)
# --------------------------------------------------------------------------

BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
BATCH_SIZE = 128

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"

# Diagnostic parameters
N_BANDS = 5
M_PROBES = 8          # minibatches per band per probe; C_k floor is ~1/M
PROBE_BS = 128
SCORE_BS = 512
EMA_BETA = 0.9        # per-sample loss EMA for the dynamic partition
DELTA_EPOCHS = 5      # horizon for the R3 predictive check


def build_datasets():
    train_tf = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_tf = Compose([
        ToTensor(),
        Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_aug = CIFAR100(root=DATA_ROOT, train=True, download=True,
                         transform=train_tf)
    train_plain = CIFAR100(root=DATA_ROOT, train=True, download=True,
                           transform=eval_tf)
    test_plain = CIFAR100(root=DATA_ROOT, train=False, download=True,
                          transform=eval_tf)
    return train_aug, train_plain, test_plain


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Per-sample scoring (forward only, unaugmented, no side effects)
# --------------------------------------------------------------------------

def score_all_samples(model, dataset, device, num_workers):
    """Per-sample loss and margin over the whole training set, unaugmented.

    Runs in eval mode. Forward only, so BN running statistics are untouched.
    Returns (loss[N], margin[N]) as float32 numpy arrays in dataset index order.
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=SCORE_BS, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    losses, margins = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(
                nn.functional.cross_entropy(logits, y, reduction="none").cpu())
            true_logit = logits.gather(1, y[:, None]).squeeze(1)
            tmp = logits.clone()
            tmp.scatter_(1, y[:, None], float("-inf"))
            margins.append((true_logit - tmp.max(dim=1).values).cpu())
    if was_training:
        model.train()
    return (torch.cat(losses).numpy().astype(np.float32),
            torch.cat(margins).numpy().astype(np.float32))


def quantile_bands(scores, n_bands):
    """Index lists for equal-count bands, band 0 = lowest score (easiest by
    loss, hardest by margin -- see caller for orientation).
    """
    order = np.argsort(scores, kind="stable")
    return [np.sort(chunk) for chunk in np.array_split(order, n_bands)]


# --------------------------------------------------------------------------
# Gradient probe (backward passes; BN statistics snapshotted and restored)
# --------------------------------------------------------------------------

def _snapshot_bn(model):
    return {k: v.detach().clone()
            for k, v in model.state_dict().items()
            if ("running_mean" in k or "running_var" in k
                or "num_batches_tracked" in k)}


def _restore_bn(model, snap):
    sd = model.state_dict()
    for k, v in snap.items():
        sd[k].copy_(v)


def _flat_grad(model, out):
    """Write the current flat gradient into `out` (preallocated)."""
    off = 0
    for p in model.parameters():
        if p.grad is None:
            n = p.numel()
            out[off:off + n].zero_()
        else:
            n = p.numel()
            out[off:off + n].copy_(p.grad.detach().reshape(-1))
        off += n
    return out


def probe_band(model, dataset_aug, indices, criterion, device, rng,
               g_sum, g_tmp):
    """Gradient statistics for one band, in TRAINING mode with augmentation,
    because the quantity of interest is the gradient signal training actually
    receives from that band.

    Returns dict with:
      mean_grad_norm  ||mean_m g_m||       coherent component
      rms_grad_norm   sqrt(mean_m ||g_m||^2)
      coherence       ||mean_m g_m||^2 / mean_m ||g_m||^2  in [~1/M, 1]
    """
    was_training = model.training
    snap = _snapshot_bn(model)
    model.train()

    g_sum.zero_()
    sum_sq = 0.0
    n_used = 0
    for _ in range(M_PROBES):
        take = min(PROBE_BS, len(indices))
        sel = rng.choice(indices, size=take, replace=False)
        xs = torch.stack([dataset_aug[int(i)][0] for i in sel]).to(device)
        ys = torch.tensor([dataset_aug[int(i)][1] for i in sel]).to(device)

        model.zero_grad(set_to_none=False)
        criterion(model(xs), ys).backward()
        _flat_grad(model, g_tmp)
        g_sum.add_(g_tmp)
        sum_sq += float(g_tmp.dot(g_tmp))
        n_used += 1

    model.zero_grad(set_to_none=True)
    _restore_bn(model, snap)
    if not was_training:
        model.eval()

    mean_g = g_sum / n_used
    mean_sq = sum_sq / n_used
    mean_norm_sq = float(mean_g.dot(mean_g))
    return {
        "mean_grad_norm": math.sqrt(max(mean_norm_sq, 0.0)),
        "rms_grad_norm": math.sqrt(max(mean_sq, 0.0)),
        "coherence": mean_norm_sq / (mean_sq + 1e-30),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

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
    set_seed(args.seed)
    nw = get_system_core_count()

    train_aug, train_plain, test_plain = build_datasets()
    train_loader = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_plain, batch_size=256, shuffle=False,
                             num_workers=nw, pin_memory=True)

    model = ResNet18(input_shape=(3, 32, 32), num_classes=100).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    if args.lr_schedule == "cosine":
        sched = CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.lr_schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _e: 1.0)
    else:  # step: 10x drops at 50% and 75% of training
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(0.5 * args.epochs), int(0.75 * args.epochs)],
            gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    g_sum = torch.zeros(n_params, device=device)
    g_tmp = torch.zeros(n_params, device=device)
    rng = np.random.default_rng(args.seed + 10_000)

    external_scores = None
    if args.static_scores:
        external_scores = np.asarray(
            json.load(open(args.static_scores)), dtype=np.float32)
        if len(external_scores) != len(train_plain):
            raise SystemExit(
                f"static scores length {len(external_scores)} != "
                f"{len(train_plain)} training samples")

    static_bands = None
    ema_loss = None

    log = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "n_bands": N_BANDS,
        "m_probes": M_PROBES,
        "probe_bs": PROBE_BS,
        "ema_beta": EMA_BETA,
        "delta_epochs": DELTA_EPOCHS,
        "dynamic_signal": args.dynamic_signal,
        "lr_schedule": args.lr_schedule,
        "static_source": (args.static_scores if args.static_scores
                          else "frozen epoch-1 unaugmented per-sample loss"),
        "coherence_floor_approx": 1.0 / M_PROBES,
        "note": "measurement only; sampling is plain shuffled throughout",
        "epochs_log": [],
    }

    for epoch in range(1, args.epochs + 1):
        eta = opt.param_groups[0]["lr"]

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

        # --- score every training sample (forward only, unaugmented) --------
        per_loss, per_margin = score_all_samples(model, train_plain, device, nw)
        ema_loss = (per_loss if ema_loss is None
                    else EMA_BETA * ema_loss + (1.0 - EMA_BETA) * per_loss)

        # --- partitions ------------------------------------------------------
        if static_bands is None:
            src = external_scores if external_scores is not None else per_loss
            static_bands = quantile_bands(src, N_BANDS)

        if args.dynamic_signal == "ema_loss":
            dyn_src = ema_loss                 # band 0 = lowest loss = easiest
        else:
            dyn_src = -per_margin              # negate so band 0 = easiest
        dynamic_bands = quantile_bands(dyn_src, N_BANDS)

        # --- per-band gradient probes ---------------------------------------
        bands_log = {}
        for pname, bands in (("static", static_bands),
                             ("dynamic", dynamic_bands)):
            entries = []
            for k, idx in enumerate(bands):
                stats = probe_band(model, train_aug, idx, criterion, device,
                                   rng, g_sum, g_tmp)
                stats["band"] = k
                stats["n"] = int(len(idx))
                stats["mean_loss"] = float(per_loss[idx].mean())
                stats["mean_ema_loss"] = float(ema_loss[idx].mean())
                stats["mean_margin"] = float(per_margin[idx].mean())
                entries.append(stats)
            bands_log[pname] = entries

        log["epochs_log"].append({
            "epoch": epoch,
            "lr": eta,
            "train_loss": run_loss / seen,
            "test_acc": test_acc,
            "bands": bands_log,
        })
        d0 = bands_log["dynamic"][0]
        d4 = bands_log["dynamic"][-1]
        print(f"ep {epoch:3d} lr={eta:.5f} loss={run_loss/seen:.4f} "
              f"test={test_acc:.2f} | dyn C: "
              f"b0={d0['coherence']:.4f} b4={d4['coherence']:.4f}")

        sched.step()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n-> {args.output_path}")


# --------------------------------------------------------------------------
# Analysis: the three pre-registered criteria
# --------------------------------------------------------------------------

def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra.dot(ra)) * float(rb.dot(rb)))
    return float(ra.dot(rb) / den) if den > 0 else float("nan")


def _activity(entries_by_epoch, k, D=None):
    """A_k(t) = Gtilde_k * C_k * P_k, assembled post hoc from logged components
    so the definition can be revised without rerunning.
      Gtilde_k = rms_grad_norm / EMA[rms_grad_norm]
      C_k      = coherence
      P_k      = max(0, EMA_loss_k(t-DELTA) - EMA_loss_k(t))
    """
    rms = np.array([e[k]["rms_grad_norm"] for e in entries_by_epoch])
    coh = np.array([e[k]["coherence"] for e in entries_by_epoch])
    eml = np.array([e[k]["mean_ema_loss"] for e in entries_by_epoch])

    ema = np.empty_like(rms)
    acc = rms[0]
    for i, v in enumerate(rms):
        acc = 0.9 * acc + 0.1 * v
        ema[i] = acc
    gt = rms / (ema + 1e-12)

    D = DELTA_EPOCHS if D is None else D
    prog = np.zeros_like(eml)
    prog[D:] = np.maximum(0.0, eml[:-D] - eml[D:])
    return gt * coh * prog, {"gtilde": gt, "coh": coh, "ema_loss": eml,
                             "progress": prog, "rms": rms}


def analyze(path):
    log = json.load(open(path))
    K = log["n_bands"]
    D = log["delta_epochs"]
    L = log["epochs_log"]
    epochs = np.array([e["epoch"] for e in L])

    print(f"file: {path}   seed={log['seed']}  bands={K}  "
          f"M={log['m_probes']}  coherence floor ~{1.0/log['m_probes']:.3f}")
    print(f"static source: {log['static_source']}")
    print(f"lr schedule: {log.get('lr_schedule', 'cosine')}")
    print(f"dynamic signal: {log['dynamic_signal']}\n")

    for pname in ("static", "dynamic"):
        ent = [e["bands"][pname] for e in L]
        A = np.stack([_activity(ent, k, D)[0] for k in range(K)])   # K x T
        comps = [_activity(ent, k, D)[1] for k in range(K)]

        print(f"=== partition: {pname} ===")

        # R1 ridge
        argmax = A.argmax(axis=0)
        i10 = int(np.argmin(np.abs(epochs - 10)))
        i70 = int(np.argmin(np.abs(epochs - 70)))
        # smooth argmax with a 5-epoch median to resist single-epoch flips
        sm = np.array([int(np.median(argmax[max(0, i - 2):i + 3]))
                       for i in range(len(argmax))])
        advance = int(sm[i70] - sm[i10])
        trend = _spearman(epochs, sm)
        r1 = (advance >= 2) and (trend >= 0)
        print(f"  R1 ridge      argmax band ep10={sm[i10]} ep70={sm[i70]} "
              f"advance={advance:+d}  trend rho={trend:+.3f}  "
              f"-> {'PASS' if r1 else 'FAIL'}")

        # R2 separation: across-band spread at a given epoch, versus each
        # band's own epoch-to-epoch noise. The within term must be measured
        # ALONG TIME per band (axis=1), then pooled across bands.
        with np.errstate(invalid="ignore"):
            across = np.nanstd(A, axis=0)                      # per epoch
            within_per_band = np.nanstd(np.diff(A, axis=1), axis=1) / math.sqrt(2)
            within = float(np.nanmean(within_per_band))         # scalar
        ratio = float(np.nanmax(across[D:]) / (within + 1e-30))
        r2 = ratio > 2.0
        print(f"  R2 separation max across-band spread / within-band noise "
              f"= {ratio:.2f}  -> {'PASS' if r2 else 'FAIL'}")

        # R3 coherence sign, computed WITHIN each epoch across bands and then
        # aggregated. Pooling all (band, epoch) pairs into one correlation is
        # confounded by the global time trend present in both quantities; the
        # controller's real question is "among the bands available right now,
        # do the more coherent ones learn more over the next D epochs?"
        C = np.stack([comps[k]["coh"] for k in range(K)])        # K x T
        E = np.stack([comps[k]["ema_loss"] for k in range(K)])   # K x T
        T = C.shape[1]
        per_epoch = []
        for t in range(T - D):
            # FRACTIONAL drop, not absolute. Bands sit at very different loss
            # levels; an easy band near zero loss cannot drop much in absolute
            # terms however well it is learning, so an absolute-drop ranking is
            # dominated by level rather than by learning.
            drop = (E[:, t] - E[:, t + D]) / (E[:, t] + 1e-8)
            if np.ptp(C[:, t]) == 0 or np.ptp(drop) == 0:
                continue
            r = _spearman(C[:, t], drop)
            if not math.isnan(r):
                per_epoch.append(r)
        if per_epoch:
            rho = float(np.mean(per_epoch))
            frac_pos = float(np.mean([r > 0 for r in per_epoch]))
        else:
            rho, frac_pos = float("nan"), float("nan")

        # The pooled figure averages opposite regimes and hides sign flips.
        pe = np.asarray(per_epoch, dtype=float)
        if len(pe) >= 9:
            n3 = len(pe) // 3
            early, mid, late = pe[:n3].mean(), pe[n3:2*n3].mean(), pe[2*n3:].mean()
            print(f"  R3 phases     early={early:+.3f}  mid={mid:+.3f}  "
                  f"late={late:+.3f}")
            w = 7
            sm3 = np.array([np.nanmean(pe[max(0, i-w//2):i+w//2+1])
                            for i in range(len(pe))])
            xing = None
            for i in range(len(sm3) - 5):
                if np.all(sm3[i:i+5] < 0):
                    xing = i + 1
                    break
            if xing is not None and early > 0:
                print(f"                sign flips at epoch ~{xing} "
                      f"(first of 5 consecutive negative, 7-epoch smoothed)")
        print(f"  R3 sign       mean within-epoch Spearman(C_k, frac loss drop "
              f"over {D} ep) = {rho:+.3f}   epochs positive = {frac_pos:.2f}   "
              f"n_epochs={len(per_epoch)}")
        if rho > 0.1:
            print("                coherence predicts MORE learning; "
                  "GAFC's term points the right way")
        elif rho < -0.1:
            print("                coherence predicts LESS learning; "
                  "steering toward coherent bands lowers diversity and "
                  "the mechanism should NOT be built as specified")
        else:
            print("                no usable relationship; coherence is not "
                  "an informative readiness signal here")

        # coherence trajectory, for eyeballing
        print("  coherence by band (epochs 1,10,30,50,70,100):")
        picks = [int(np.argmin(np.abs(epochs - e)))
                 for e in (1, 10, 30, 50, 70, 100) if e <= epochs[-1]]
        hdr = "    band " + "".join(f"{epochs[i]:>9d}" for i in picks)
        print(hdr)
        for k in range(K):
            row = "".join(f"{comps[k]['coh'][i]:9.4f}" for i in picks)
            print(f"    {k:<5}{row}")
        print()

    print("Interpretation: R1 and R2 both failing means no exploitable "
          "frontier. R3 negative means the frontier may exist but coherence "
          "is the wrong readiness signal.")
    print("These statistics are computed by this script ad hoc; rerun them in "
          "your own analysis code before citing.")


def main():
    global M_PROBES, DELTA_EPOCHS

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--output-path", type=str)
    ap.add_argument("--static-scores", type=str, default=None,
                    help="JSON list of per-sample scores in dataset index "
                         "order; default freezes epoch-1 per-sample loss")
    ap.add_argument("--dynamic-signal", choices=["ema_loss", "margin"],
                    default="ema_loss")
    ap.add_argument("--m-probes", type=int, default=M_PROBES,
                    help="minibatch gradients per band per probe; the raw "
                         "coherence statistic has a floor of ~1/M, so raise "
                         "this to resolve late-training coherence")
    ap.add_argument("--delta-epochs", type=int, default=DELTA_EPOCHS,
                    help="prediction horizon for the R3 check")
    ap.add_argument("--lr-schedule", choices=["cosine", "constant", "step"],
                    default="cosine",
                    help="cosine is the established regime. Under cosine the "
                         "LR is a function of t/T alone, so training progress "
                         "and LR level are perfectly confounded; 'constant' "
                         "breaks that confound (at the cost of final accuracy, "
                         "which does not matter for a measurement run). 'step' "
                         "drops by 10x at 50%% and 75%% of training.")
    ap.add_argument("--analyze", type=str, default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    if args.seed is None or args.output_path is None:
        ap.error("--seed and --output-path are required")

    M_PROBES = args.m_probes
    DELTA_EPOCHS = args.delta_epochs
    run(args)


if __name__ == "__main__":
    main()