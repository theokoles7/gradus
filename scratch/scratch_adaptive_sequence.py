#!/usr/bin/env python3
"""
scratch_adaptive_sequence.py

Holding the minibatch partition fixed, can model-state-dependent sequence
selection improve over random minibatch ordering?

    Control:  ordinary random batches + random order
    Policy:   ordinary random batches + adaptively chosen order

No difficulty metric, no homogeneous batches, no composition penalty, no
easy/hard assumption. The baseline IS plain shuffled SGD, not a proxy for it,
so a win is stated directly rather than through an intermediate arm.

Why this is the experiment
--------------------------
Established (3 seeds, 100 epochs, saturation-time): the holistic curriculum
penalty is -3.47pp, of which -2.66pp (77%) is presentation ORDER and only
-0.82pp is batch composition. Order is a large lever. Monotonic order is bad,
random order is good, and nothing else has been tested.

Static permutations over a difficulty axis were considered and skipped: the
final algorithm does not use a difficulty metric, so a result there would be on
a lever this algorithm discards.

Mechanics
---------
Each epoch the universe is partitioned into nb ordinary random batches. That
partition is then FIXED for the epoch. Batches are presented one at a time:
a candidate pool of K unconsumed batches is maintained, each candidate is
scored, the best is consumed, and the pool is refilled. Every batch is consumed
exactly once, so coverage, batch size, and gradient-step count are identical to
the control by construction.

Selection signals (--policy)
----------------------------
  random        pop uniformly from the pool. THE CONTROL. Zero overhead. With
                a random partition this is exactly ordinary shuffled SGD.

  progress_high pick the batch whose samples showed the largest decrease in
                EMA loss the LAST TIME they were consumed. The per-sample
                progress array is written during ordinary training, so this
                costs NOTHING extra. The state is from the previous epoch for
                samples not yet seen this epoch, which is the price of free.

                (An earlier version scored prev_ema_loss[idx] - ema_loss[idx]
                against a snapshot taken at epoch start. An unconsumed sample's
                ema_loss has not moved since that snapshot, so every candidate
                scored exactly 0 and argmax always returned index 0: the policy
                silently reduced to random. The progress array fixes this by
                recording the delta at consumption time.)
  progress_low  the inverse. Included because the sign is an open question,
                not a foregone conclusion.

  loss_high     pick the highest current mean loss, measured by a forward pass
                on each candidate at the current theta. Costs K forwards per
                step.
  loss_low      the inverse.

  antialign     RESIDUAL-ACCUMULATION MINIMISATION. The frozen-theta
                diagnostic was about residuals eps_t = g_t - g_ref and their
                momentum accumulation r_t = mu*r_{t-1} + eps_t, not about raw
                gradients. At each step, with candidate gradients g_1..g_K:

                    gbar   = (1/K) sum_j g_j          local estimate of the
                                                      common descent direction
                    eps_j  = g_j - gbar               candidate-specific residual
                    j*     = argmin_j || mu*r + eps_j ||
                    r      <- mu*r + eps_{j*}

                It asks directly: which available batch produces the least
                accumulated coherent residual if taken next? That is the
                quantity that became pathological under ordered curricula.

                (An earlier version kept an EMA of RAW classifier gradients and
                picked the candidate least aligned with it, i.e. -cos(g_j,
                EMA(g)). Every useful gradient shares a component of the true
                descent direction, so that policy selected against exactly the
                component all batches should have in common. It was not the
                mechanism-derived policy and is not what this implements.)

                Note: centering on the pool mean forces sum_j eps_j = 0, so the
                K scores are not independent, and the pool shrinks at the end of
                an epoch which changes the centering. An EMA-based reference is
                logged alongside for comparison.

                Costs K forwards per step (the classifier gradient is formed
                analytically from the same forward, no backward).

                CAVEAT: the classifier-only gradient was validated against
                full-parameter batch-gradient geometry and FAILED (Pearson
                +0.11 to +0.50 across checkpoints, against a 0.7 bar). So this
                signal is a classifier-layer quantity that may not represent
                the network's gradient geometry. It is cheap and worth
                screening; a positive result would need re-running with full
                gradients before it means what it says.

Cost
----
  random, progress_*   1.00x (no extra forward passes)
  loss_*, antialign    roughly 1 + K/3/select_every times a plain run, since a
                       forward is about a third of a training step. K=4 every
                       step is ~2.3x; --select-every 2 halves the overhead by
                       falling back to random on off steps.

Candidate scoring runs in eval mode under no_grad, so BatchNorm running
statistics are never updated by scoring.

Usage
-----
    python scratch_adaptive_sequence.py --self-test

    python scratch_adaptive_sequence.py --seed 1 --policy random \\
        --output-path results/adaptive/adapt_random_seed1.json
    python scratch_adaptive_sequence.py --seed 1 --policy antialign --pool 4 \\
        --output-path results/adaptive/adapt_antialign_seed1.json

    python scratch_adaptive_sequence.py --analyze results/adaptive/*.json
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
CosineAnnealingLR = None
CIFAR100 = None
Compose = Normalize = RandomCrop = RandomHorizontalFlip = ToTensor = None
ResNet18 = None
get_system_core_count = None


def _lazy_imports():
    global torch, nn, F, DataLoader, CosineAnnealingLR, CIFAR100
    global Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
    global ResNet18, get_system_core_count

    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.utils.data import DataLoader as _DL
    from torch.optim.lr_scheduler import CosineAnnealingLR as _Cos
    from torchvision.datasets import CIFAR100 as _C
    from torchvision.transforms import (Compose as _Cp, Normalize as _Nm,
                                        RandomCrop as _RC,
                                        RandomHorizontalFlip as _RH,
                                        ToTensor as _TT)
    # RECONCILE: confirm against the installed package layout.
    from gradus.networks.resnet.resnet_18 import ResNet18 as _R
    from gradus.utilities import get_system_core_count as _c

    torch, nn, F, DataLoader, CosineAnnealingLR, CIFAR100 = (
        _t, _nn, _F, _DL, _Cos, _C)
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor = (
        _Cp, _Nm, _RC, _RH, _TT)
    ResNet18, get_system_core_count = _R, _c


BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
NUM_CLASSES = 100
EPOCHS = 100

EMA_BETA = 0.9          # per-sample loss EMA
DIR_EMA = 0.9           # EMA of recent classifier update direction
POOL = 4

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"

FREE_POLICIES = ("random", "progress_high", "progress_low")
FWD_POLICIES = ("loss_high", "loss_low", "antialign")
POLICIES = FREE_POLICIES + FWD_POLICIES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets():
    """Three views of the data.

    ds_train   augmented; used for the actual gradient steps.
    ds_score   UNAUGMENTED view of the same training set, used only for
               candidate scoring. Scoring through ds_train would draw a fresh
               RandomCrop/Flip for every candidate sample, so a pool-4 decision
               would consume ~512 augmentation draws before the real batch is
               even loaded. The forward-based arms would then differ from the
               control in their augmentation stream as well as in sequence,
               which is not the comparison being made.
    ds_test    evaluation.
    """
    train_tf = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(),
                        ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    eval_tf = Compose([ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    return (CIFAR100(root=DATA_ROOT, train=True, download=True,
                     transform=train_tf),
            CIFAR100(root=DATA_ROOT, train=True, download=True,
                     transform=eval_tf),
            CIFAR100(root=DATA_ROOT, train=False, download=True,
                     transform=eval_tf))


def random_partition(universe, bs, rng):
    """Ordinary random batches over the WHOLE training set, with a short final
    batch, matching a shuffled DataLoader at drop_last=False.

    Dropping the remainder would train on 49,920 of 50,000 samples in 390 steps
    and the control would then not be the established shuffled baseline, which
    takes 391 steps. Loss and classifier-gradient calculations normalise by the
    actual batch size, so the short final batch is harmless.
    """
    u = np.asarray(universe)
    o = u[rng.permutation(len(u))]
    return [o[t:t + bs] for t in range(0, len(o), bs)]


class FeatureTap:
    def __init__(self, model):
        self.h = None
        # RECONCILE: Gradus names the classifier _fc_.
        self._handle = model._fc_.register_forward_hook(self._hook)

    def _hook(self, _m, inp, _out):
        self.h = inp[0].detach()

    def close(self):
        self._handle.remove()


def classifier_grad(logits, targets, h):
    """Exact classifier-layer gradient from a forward pass, flattened.

    For cross-entropy, dL/dW = (1/n) (P - Y)^T H and dL/db = mean(P - Y).
    No backward pass required.
    """
    p = torch.softmax(logits.float(), dim=1)
    y = torch.zeros_like(p).scatter_(1, targets[:, None], 1.0)
    d = p - y
    gW = d.t().mm(h.float()) / d.shape[0]
    gb = d.mean(0)
    return torch.cat([gW.reshape(-1), gb.reshape(-1)])


class Selector:
    """Chooses the presentation order within an epoch.

    Maintains a pool of K unconsumed batches, scores them, consumes the best.
    Every batch is consumed exactly once, so coverage and step count match the
    control exactly.
    """

    def __init__(self, policy, pool, select_every, device,
                 grad_repr="classifier"):
        self.grad_repr = grad_repr
        self.policy = policy
        self.pool = pool
        self.select_every = select_every
        self.device = device
        self.r = None                # residual accumulator: r = mu*r + eps
        self.g_ema = None            # EMA of batch gradients (reference, logged)
        self._eps = None             # residuals of the current pool
        self.n_scored = 0
        self.n_forward = 0
        self.r_norms = []

    def needs_forward(self):
        return self.policy in FWD_POLICIES

    def score(self, model, dataset, cand, progress, tap, criterion):
        """Higher score = selected. One score per candidate batch."""
        if self.policy == "random":
            return None

        if self.policy in ("progress_high", "progress_low"):
            # free: per-sample EMA-loss decrease recorded at last consumption
            s = np.array([float(progress[idx].mean()) for idx in cand])
            return s if self.policy == "progress_high" else -s

        # forward-based policies
        was_training = model.training
        model.eval()                     # no BN running-stat updates
        scores, grads = [], []
        for idx in cand:
            self.n_forward += 1
            if self.policy in ("loss_high", "loss_low"):
                with torch.no_grad():
                    xs = torch.stack([dataset[int(i)][0]
                                      for i in idx]).to(self.device)
                    ys = torch.tensor([dataset[int(i)][1]
                                       for i in idx]).to(self.device)
                    scores.append(float(criterion(model(xs), ys)))
            else:
                grads.append(candidate_gradient(model, dataset, idx, criterion,
                                                self.device, self.grad_repr,
                                                tap))
        if was_training:
            model.train()

        if self.policy == "loss_high":
            return np.array(scores)
        if self.policy == "loss_low":
            return -np.array(scores)

        # antialign: minimise the accumulated coherent residual
        G = torch.stack(grads)                       # K x D
        gbar = G.mean(0)                             # common descent estimate
        eps = G - gbar                               # sum_j eps_j = 0
        self._eps = eps
        if self.r is None:
            self.r = torch.zeros_like(gbar)
        base = self.r * MOMENTUM
        sc = np.array([-float((base + eps[j]).norm()) for j in range(len(grads))])
        return sc

    def commit(self, pick):
        """Fold the chosen candidate's residual into the accumulator."""
        if self._eps is None:
            return
        self.r.mul_(MOMENTUM).add_(self._eps[pick])
        self.r_norms.append(float(self.r.norm()))
        self._eps = None

    def note_gradient(self, g):
        """EMA of batch gradients; an alternative reference, logged only."""
        if self.g_ema is None:
            self.g_ema = g.clone()
        else:
            self.g_ema.mul_(DIR_EMA).add_(g, alpha=1.0 - DIR_EMA)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total



def _repr_params(model, repr_name):
    """Parameters whose gradient defines the candidate representation.

    classifier  the final linear layer only. Analytic from the forward pass,
                no backward. Cheapest, and it FAILED decision validation
                (top-1 0.335 against a 0.25 chance floor at K=4).
    layer4      the last residual stage plus the classifier. autograd stops at
                layer4, so the backward traverses one block rather than the
                whole network: much cheaper than `full` while covering 8.4M of
                11.2M parameters.
    full        every parameter. Most faithful, most expensive.
    """
    if repr_name == "classifier":
        return list(model._fc_.parameters())
    if repr_name == "layer4":
        return list(model._layer4_.parameters()) + list(model._fc_.parameters())
    if repr_name == "full":
        return [p for p in model.parameters() if p.requires_grad]
    raise ValueError(repr_name)


def candidate_gradient(model, dataset, idx, criterion, device, repr_name, tap):
    """One batch's gradient under the chosen representation, at fixed theta."""
    if repr_name == "classifier":
        with torch.no_grad():
            xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
            ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
            logits = model(xs)
            return classifier_grad(logits, ys, tap.h)

    params = _repr_params(model, repr_name)
    xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
    ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
    loss = criterion(model(xs), ys)
    gs = torch.autograd.grad(loss, params, retain_graph=False)
    return torch.cat([g.detach().reshape(-1) for g in gs])


def full_classifier_and_network_grads(model, dataset, idx, criterion, device,
                                      n_params):
    """Both representations of one batch's gradient, at the same theta.

    Returns (classifier_grad, full_parameter_grad). The full version needs a
    real backward pass; the classifier version is analytic from the forward.
    """
    tap = FeatureTap(model)
    xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
    ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
    model.zero_grad(set_to_none=False)
    logits = model(xs)
    criterion(logits, ys).backward()
    clf = classifier_grad(logits.detach(), ys, tap.h).detach().clone()
    full = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel(), device=device))
                      for p in model.parameters()]).clone()
    model.zero_grad(set_to_none=True)
    tap.close()
    return clf, full


def _residual_scores(grads, r):
    """argmin_j ||mu*r + eps_j||, with eps centered on the pool mean."""
    G = torch.stack(grads)
    eps = G - G.mean(0)
    base = r * MOMENTUM
    sc = np.array([-float((base + eps[j]).norm()) for j in range(len(grads))])
    return sc, eps


def validate_signal(args):
    """Does the cheap classifier proxy make the SAME selection as the full
    gradient?

    The earlier Pearson test asked whether classifier geometry globally
    resembles full-network geometry (it does not: +0.11 to +0.50). But the
    controller only needs the proxy to pick the same batch. With K candidates,
    chance top-1 agreement is 1/K, so that is the number to beat.

    Runs ordinary random-order training and, every --validate-every steps,
    scores the pool under both representations with independent residual
    accumulators, recording whether the argmin agrees.
    """
    _lazy_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    nw = get_system_core_count()
    ds_train, ds_score, ds_test = build_datasets()

    n_total = len(ds_train)
    universe = np.arange(n_total)
    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    brng = np.random.default_rng(args.seed * 1000 + 7)
    tap_v = FeatureTap(model)

    r_clf = None
    r_full = None
    top1, top2, rhos, checks = [], [], [], 0

    def sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = math.sqrt(float(ra.dot(ra)) * float(rb.dot(rb)))
        return float(ra.dot(rb) / d) if d > 0 else float("nan")

    step_global = 0
    for epoch in range(1, args.epochs + 1):
        batches = random_partition(universe, BATCH_SIZE, brng)
        remaining = list(range(len(batches)))
        brng.shuffle(remaining)
        model.train()
        while remaining:
            k = min(args.pool, len(remaining))
            pool_ids = remaining[:k]

            if (step_global % args.validate_every == 0 and k == args.pool
                    and checks < args.validate_checks):
                was = model.training
                model.eval()
                cl, fu = [], []
                for i in pool_ids:
                    cl.append(candidate_gradient(model, ds_score, batches[i],
                                                 criterion, device,
                                                 args.grad_repr, tap_v))
                    fu.append(candidate_gradient(model, ds_score, batches[i],
                                                 criterion, device,
                                                 "full", tap_v))
                if was:
                    model.train()
                if r_clf is None:
                    r_clf = torch.zeros_like(cl[0])
                    r_full = torch.zeros_like(fu[0])
                s_clf, e_clf = _residual_scores(cl, r_clf)
                s_full, e_full = _residual_scores(fu, r_full)
                pc, pf = int(np.argmax(s_clf)), int(np.argmax(s_full))
                top1.append(pc == pf)
                top2.append(pc in np.argsort(-s_full)[:2])
                rhos.append(sp(s_clf, s_full))
                # both accumulators follow the FULL-gradient decision, so the
                # trajectories stay comparable
                r_clf = r_clf * MOMENTUM + e_clf[pf]
                r_full = r_full * MOMENTUM + e_full[pf]
                checks += 1

            chosen = pool_ids[0]
            remaining.remove(chosen)
            idx = batches[chosen]
            xs = torch.stack([ds_train[int(i)][0] for i in idx]).to(device)
            ys = torch.tensor([ds_train[int(i)][1] for i in idx]).to(device)
            opt.zero_grad(set_to_none=True)
            criterion(model(xs), ys).backward()
            opt.step()
            step_global += 1
        sched.step()
        if checks >= args.validate_checks:
            break

    n = len(top1)
    chance = 1.0 / args.pool
    print(f"\nsignal validation: K={args.pool}, {n} checks over "
          f"{epoch} epoch(s)")
    print(f"  top-1 agreement (classifier vs full): {np.mean(top1):.3f}  "
          f"(chance {chance:.3f})")
    print(f"  top-2 containment:                    {np.mean(top2):.3f}  "
          f"(chance {2*chance:.3f})")
    print(f"  mean Spearman of the {args.pool} candidate scores: "
          f"{np.nanmean(rhos):+.3f}")
    verdict = ("proxy tracks the full-gradient decision; proceed to 9 seeds"
               if np.mean(top1) >= 0.55 else
               "proxy does NOT track the decision; upgrade the gradient "
               "representation (classifier -> layer4+classifier -> full) "
               "before spending seeds")
    print(f"  verdict: {verdict}")
    out = {"seed": args.seed, "pool": args.pool, "grad_repr": args.grad_repr,
           "n_checks": n,
           "top1_agreement": float(np.mean(top1)),
           "top2_containment": float(np.mean(top2)),
           "mean_spearman": float(np.nanmean(rhos)), "chance_top1": chance}
    if args.output_path:
        d = os.path.dirname(args.output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> {args.output_path}")


def run(args):
    _lazy_imports()
    if os.path.exists(args.output_path):
        raise SystemExit(f"{args.output_path} exists. Use a distinct path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = get_system_core_count()
    ds_train, ds_score, ds_test = build_datasets()
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                             num_workers=nw, pin_memory=True)

    n_total = len(ds_train)
    universe = np.arange(n_total)

    set_seed(args.seed)
    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    tap = FeatureTap(model)

    brng = np.random.default_rng(args.seed * 1000 + 7)
    sel = Selector(args.policy, args.pool, args.select_every, device,
                   grad_repr=args.grad_repr)

    ema_loss = np.zeros(n_total, dtype=np.float32)
    progress = np.zeros(n_total, dtype=np.float32)   # last observed EMA drop
    seen_once = np.zeros(n_total, dtype=bool)

    log, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        batches = random_partition(universe, BATCH_SIZE, brng)
        remaining = list(range(len(batches)))
        brng.shuffle(remaining)

        model.train()
        rl, seen, step = 0.0, 0, 0
        while remaining:
            k = min(sel.pool, len(remaining))
            pool_ids = remaining[:k]
            cand = [batches[i] for i in pool_ids]

            use_policy = (args.policy != "random"
                          and step % args.select_every == 0)
            if use_policy:
                s = sel.score(model, ds_score, cand, progress, tap, criterion)
                pick = int(np.argmax(s))
                sel.commit(pick)
                sel.n_scored += 1
            else:
                pick = 0                 # remaining is already shuffled
            chosen = pool_ids[pick]
            remaining.remove(chosen)
            idx = batches[chosen]

            xs = torch.stack([ds_train[int(i)][0] for i in idx]).to(device)
            ys = torch.tensor([ds_train[int(i)][1] for i in idx]).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xs)
            per = F.cross_entropy(logits, ys, reduction="none")
            loss = per.mean()
            loss.backward()

            if sel.policy == "antialign" and sel.grad_repr == "classifier":
                with torch.no_grad():
                    sel.note_gradient(classifier_grad(logits.detach(), ys,
                                                      tap.h).detach())
            opt.step()

            pl = per.detach().cpu().numpy()
            fresh = ~seen_once[idx]
            old = ema_loss[idx].copy()
            new = np.where(fresh, pl, EMA_BETA * old + (1 - EMA_BETA) * pl)
            progress[idx] = np.where(fresh, 0.0, old - new)
            ema_loss[idx] = new
            seen_once[idx] = True

            rl += float(loss) * len(idx)
            seen += len(idx)
            step += 1

        acc = evaluate(model, test_loader, device)
        best = max(best, acc)
        log.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                    "train_loss": rl / seen, "test_acc": acc})
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"ep {epoch:3d} loss={rl/seen:.4f} test={acc:.2f} "
                  f"best={best:.2f}")
        sched.step()

    tap.close()
    out = {"seed": args.seed, "policy": args.policy, "epochs": args.epochs,
           "batch_size": BATCH_SIZE, "pool": args.pool,
           "select_every": args.select_every, "n_batches": len(batches),
           "ema_beta": EMA_BETA, "dir_ema": DIR_EMA,
           "grad_repr": args.grad_repr,
           "selection_steps": sel.n_scored, "scoring_forwards": sel.n_forward,
           "residual_norm_mean": (float(np.mean(sel.r_norms))
                                  if sel.r_norms else None),
           "residual_norm_max": (float(np.max(sel.r_norms))
                                 if sel.r_norms else None),
           "note": "ordinary random batches over all 50,000 samples, fixed "
                   "per epoch; only the presentation order differs between "
                   "policies. Candidate scoring uses an unaugmented view so it "
                   "does not consume training augmentation draws.",
           "best_test_acc": best, "final_test_acc": log[-1]["test_acc"],
           "epochs_log": log}
    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\npolicy={args.policy} best={best:.2f} final={log[-1]['test_acc']:.2f}")
    print(f"-> {args.output_path}")


def self_test():
    """Verify selection mechanics without torch: exactly-once consumption, full
    coverage of all 50,000 samples, 391 steps, and that the progress signal
    actually differentiates candidates (an earlier version did not).
    """
    N, bs = 50000, 128
    rng = np.random.default_rng(0)
    U = np.arange(N)
    B = random_partition(U, bs, rng)
    nb = len(B)
    sizes = sorted({len(b) for b in B})
    print(f"{nb} random batches over {N} samples; batch sizes {sizes}")
    flat = np.sort(np.concatenate(B))
    print(f"  covers universe exactly: {np.array_equal(flat, U)}")
    print(f"  steps per epoch: {nb} (ordinary shuffled DataLoader: "
          f"{math.ceil(N / bs)})")
    assert np.array_equal(flat, U) and nb == math.ceil(N / bs)

    for pool in (1, 4, 16):
        remaining = list(range(nb))
        rng2 = np.random.default_rng(1)
        rng2.shuffle(remaining)
        consumed, positions, step = [], [], 0
        while remaining:
            k = min(pool, len(remaining))
            ids = remaining[:k]
            pick = int(rng2.integers(k))
            positions.append(pick)
            chosen = ids[pick]
            remaining.remove(chosen)
            consumed.append(chosen)
            step += 1
        ok = sorted(consumed) == list(range(nb))
        print(f"  pool={pool:<3} steps={step} each batch once={ok} "
              f"max pick index={max(positions)}")
        assert ok and step == nb

    # progress signal must differentiate candidates, not return all zeros
    progress = np.zeros(N, dtype=np.float32)
    rng3 = np.random.default_rng(2)
    touched = rng3.choice(N, size=N // 2, replace=False)
    progress[touched] = rng3.normal(size=len(touched)).astype(np.float32)
    cand = B[:4]
    sc = np.array([float(progress[idx].mean()) for idx in cand])
    print(f"\n  progress scores over a 4-batch pool: "
          f"{' '.join(f'{v:+.4f}' for v in sc)}")
    print(f"  all identical (the old bug): {bool(np.ptp(sc) == 0)}")
    assert np.ptp(sc) > 0

    print("\n  free vs forward-based policies:")
    for p_ in POLICIES:
        kind = ("free (0 extra forwards)" if p_ in FREE_POLICIES
                else "forward-based (K per selection step, unaugmented view)")
        print(f"    {p_:<15} {kind}")
    print("\nself-test passed")


def analyze(paths):
    logs = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            logs.append(json.load(open(f)))
    if not logs:
        raise SystemExit("no files matched")

    by = defaultdict(dict)
    for L in logs:
        by[L["policy"]][L["seed"]] = L["best_test_acc"]
    policies = [p for p in POLICIES if p in by]
    seeds = sorted(set.intersection(*[set(by[p]) for p in policies]))
    print(f"{len(seeds)} shared seed(s): {seeds}; "
          f"{logs[0]['epochs']} epochs, pool={logs[0]['pool']}\n")

    print(f"{'seed':<6}" + "".join(f"{p:>16}" for p in policies))
    for s in seeds:
        print(f"{s:<6}" + "".join(f"{by[p][s]:>16.2f}" for p in policies))

    if "random" not in by:
        print("\nno `random` control present; cannot form paired deltas")
        return
    n = len(seeds)
    ctrl = np.array([by["random"][s] for s in seeds])
    print("\nvs random control (identical partition distribution, order only):")
    for p in policies:
        if p == "random":
            continue
        d = np.array([by[p][s] for s in seeds]) - ctrl
        m = d.mean()
        if n > 1:
            sd = d.std(ddof=1)
            se = sd / math.sqrt(n)
            print(f"  {p:<16} mean={m:+7.3f}pp  sd={sd:.3f}  "
                  f"t={m/se:+.2f} (df={n-1})  positive={int((d>0).sum())}/{n}")
        else:
            print(f"  {p:<16} mean={m:+7.3f}pp  (single seed)")

    print("\nThresholds are descriptive, not tests. These statistics are "
          "computed ad hoc by this script; rerun in your own analysis code "
          "before citing.")
    print("A positive antialign result rests on a classifier-layer gradient "
          "proxy that FAILED validation against full-parameter geometry "
          "(Pearson +0.11 to +0.50); re-run with full gradients before "
          "claiming it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--policy", choices=POLICIES, default="random")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--pool", type=int, default=POOL,
                    help="candidate pool size K; 1 reduces to random")
    ap.add_argument("--select-every", type=int, default=1,
                    help="apply the policy every W steps, random otherwise; "
                         "cuts scoring cost by W")
    ap.add_argument("--output-path", type=str)
    ap.add_argument("--analyze", nargs="+", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--validate-signal", action="store_true",
                    help="measure whether the classifier proxy makes the same "
                         "selection as the full gradient, before spending seeds")
    ap.add_argument("--validate-every", type=int, default=20)
    ap.add_argument("--validate-checks", type=int, default=200)
    ap.add_argument("--grad-repr", choices=["classifier", "layer4", "full"],
                    default="classifier",
                    help="gradient representation for antialign. classifier is "
                         "analytic and free but FAILED decision validation "
                         "(top-1 0.335 vs 0.25 chance); layer4 backprops "
                         "through the last block only; full is exact.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.analyze:
        analyze(args.analyze)
        return
    if args.validate_signal:
        if args.seed is None:
            ap.error("--seed is required")
        validate_signal(args)
        return
    if args.seed is None or args.output_path is None:
        ap.error("--seed and --output-path are required")
    run(args)


if __name__ == "__main__":
    main()