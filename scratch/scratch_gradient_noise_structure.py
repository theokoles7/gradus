#!/usr/bin/env python3
"""
scratch_gradient_noise_structure.py   (v2)

MEASUREMENT ONLY. Trains one plain shuffled run and, at fixed checkpoints,
freezes theta and evaluates several batch-construction schemes at the SAME
model state. No scheme ever updates the model.

Question
--------
What property of shuffled SGD does holistic curriculum destroy?

At fixed theta, write   ghat_t = g + eps_t,   g = (1/N) sum_i g_i(theta).

  (A) WITHIN-BATCH HOMOGENEITY: difficulty-grouped examples have more similar
      gradients, so the batch loses internal cancellation.
  (B) BETWEEN-BATCH TEMPORAL CORRELATION: consecutive curriculum batches sit
      in nearby difficulty regions, so errors accumulate as drift rather than
      cancelling.

Design corrections in v2 (all of these mattered)
-----------------------------------------------
1. IDENTICAL SAMPLE UNIVERSE. Every scheme partitions exactly the same
   nb*bs samples, chosen once. v1 truncated the batch LIST after permuting it,
   so holistic_asc[:200] was the easiest 200 batches while holistic_bshuf[:200]
   was a random 200 -- the two stopped being comparable precisely where the
   comparison mattered.

2. FINAL DRIFT CANNOT MEASURE ORDER. If two schemes hold identical batch
   vectors, sum_t eps_{pi(t)} = sum_t eps_t, so D_m is permutation invariant.
   v1 used final D_m for the ordering verdict, which is mathematically
   incapable of answering it. v2 uses order-sensitive statistics only:
     - prefix drift trajectory (mean and max of D_t / (sqrt(t)*mean||eps||))
     - drift AUC
     - lag-1..L error cosine autocorrelation
     - MOMENTUM-FILTERED error  r_t = mu*r_{t-1} + eps_t  (mu = 0.9, matching
       the optimizer), reporting ||r_t|| mean, max and AUC. Persistent
       correlated error accumulates through the momentum buffer; randomly
       ordered error cancels. This is the closest proxy to what SGD actually
       experiences.
   D_final is still logged, and is used as a CORRECTNESS CHECK: asc and bshuf
   must agree to numerical tolerance. If they do not, the universes differ.

3. NO AUGMENTATION CONFOUND. v1 took g_ref unaugmented and scheme gradients
   augmented, with a fresh random crop/flip per fetch, so each scheme saw
   different transforms of the same examples. v2 uses the unaugmented dataset
   for g_ref and for every scheme. Augmentation can be reintroduced later, but
   only with frozen pre-generated transforms shared across schemes.

4. EXPLICIT BATCHNORM MODE. v1 inherited eval mode by accident from the
   validation loop. v2 makes it a flag. eval (default) freezes BN running
   statistics and isolates pure gradient geometry; train lets homogeneous
   batches produce their own BN statistics, which is itself a candidate
   mechanism, and snapshots/restores the running stats so the diagnostic
   leaves no trace. Report which was used.

5. DIFFICULTY SOURCE IS EXPLICIT. Default is current per-sample loss, which
   answers "what happens when batches are structured by current loss
   difficulty". That is NOT the same as the original holistic curricula
   (saturation-time, edge-density, wavelet-entropy, compression-ratio, ...).
   Pass --static-scores to rerun with an actual original metric before
   claiming this explains those results.

Scheme structure (verified on synthetic scores)
-----------------------------------------------
  scheme            within-batch difficulty   between-batch difficulty-mean
  ----------------  ------------------------  ----------------------------
  shuffled                  high                      low
  fixed                     high                      low   (dataset order)
  holistic_asc              LOW                       HIGH, monotone
  holistic_desc             LOW                       HIGH, monotone
  holistic_bshuf            LOW                       HIGH, order randomised
  stride                    high                      very low

holistic_bshuf holds literally identical batch contents to holistic_asc and
differs only in presentation order. stride is described by its construction
(high within-batch difficulty spread, near-zero between-batch mean variance);
whether its gradient errors are temporally correlated is for the experiment to
measure, not to assume.

The two principal comparisons
-----------------------------
  shuffled  vs  holistic_bshuf   -> within-batch homogeneity (A), order random
                                    in both
  holistic_bshuf vs holistic_asc -> temporal ordering (B), identical batch
                                    vectors in different orders

Stated limitation
-----------------
Freezing theta answers "what does this batching do to gradient noise at this
model state", not "what does it do over 100 epochs". The error process is not
stationary in real training. This is a deliberate simplification.

Usage
-----
    python scratch_gradient_noise_structure.py --seed 1 \
        --output-path results/noise/noise_seed1.json
    python scratch_gradient_noise_structure.py --analyze \
        results/noise/noise_seed1.json
"""

import argparse
import json
import math
import os
import random

import numpy as np

# torch/torchvision/gradus imported lazily so --analyze runs without them.
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


# --------------------------------------------------------------------------
# Regime
# --------------------------------------------------------------------------

BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
BATCH_SIZE = 128
NUM_CLASSES = 100

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
DATA_ROOT = ".cache/data"

CHECKPOINTS = (1, 10, 25, 50, 75)
SCHEMES = ("shuffled", "fixed", "holistic_asc", "holistic_desc",
           "holistic_bshuf", "stride")
MAX_BATCHES = 200
MAX_LAG = 5


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
# Batch construction over a FIXED sample universe
# --------------------------------------------------------------------------

def choose_universe(n_total, bs, max_batches, rng):
    """Pick the sample set every scheme will partition, once.

    Returns an index array of length exactly nb*bs. Chosen uniformly at random
    so that truncation never correlates with difficulty -- the v1 bug was
    truncating the batch list instead, which handed holistic_asc the easiest
    batches and holistic_bshuf a random selection.
    """
    nb = n_total // bs
    if max_batches is not None:
        nb = min(nb, max_batches)
    return np.sort(rng.choice(n_total, size=nb * bs, replace=False))


def build_batches(scheme, universe, scores, bs, rng):
    """Partition `universe` into batches. scores is indexed by GLOBAL sample
    index; only scores[universe] is used. Every scheme returns the same number
    of batches covering the same samples.
    """
    u = np.asarray(universe)
    nb = len(u) // bs
    s = scores[u]

    if scheme == "shuffled":
        o = u[rng.permutation(len(u))]
        return [o[t * bs:(t + 1) * bs] for t in range(nb)]

    if scheme == "fixed":
        o = u  # already ascending global index = dataset order
        return [o[t * bs:(t + 1) * bs] for t in range(nb)]

    srt = u[np.argsort(s, kind="stable")]

    if scheme == "holistic_asc":
        return [srt[t * bs:(t + 1) * bs] for t in range(nb)]

    if scheme == "holistic_desc":
        rev = srt[::-1].copy()
        return [rev[t * bs:(t + 1) * bs] for t in range(nb)]

    if scheme == "holistic_bshuf":
        b = [srt[t * bs:(t + 1) * bs] for t in range(nb)]
        return [b[i] for i in rng.permutation(nb)]

    if scheme == "stride":
        return [srt[j:nb * bs:nb] for j in range(nb)]

    raise ValueError(scheme)


# --------------------------------------------------------------------------
# Gradient machinery at frozen theta
# --------------------------------------------------------------------------

def _flat_grad(model, out):
    off = 0
    for p in model.parameters():
        n = p.numel()
        if p.grad is None:
            out[off:off + n].zero_()
        else:
            out[off:off + n].copy_(p.grad.detach().reshape(-1))
        off += n
    return out


def _snapshot_bn(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if ("running_mean" in k or "running_var" in k
                or "num_batches_tracked" in k)}


def _restore_bn(model, snap):
    sd = model.state_dict()
    for k, v in snap.items():
        sd[k].copy_(v)


class FeatureTap:
    """Captures the classifier input so exact per-example classifier gradients
    can be formed without a backward pass per example.
    """

    def __init__(self, model):
        self.h = None
        # RECONCILE: Gradus names the classifier _fc_.
        self._handle = model._fc_.register_forward_hook(self._hook)

    def _hook(self, _m, inp, _out):
        self.h = inp[0].detach()

    def close(self):
        self._handle.remove()


def per_example_classifier_coherence(logits, targets, h):
    """||mean_i g_i||^2 / mean_i ||g_i||^2 over exact per-example classifier
    gradients. For cross-entropy, dL_i/dW = (p_i - onehot_i) outer h_i and
    dL_i/db = (p_i - onehot_i).

    NOTE: this is a classifier-layer quantity. --validate-proxy checks that
    BATCH-gradient cosines agree between classifier-only and full parameters;
    it does not directly validate that per-example coherence agrees. Treat this
    column as a classifier-layer measurement, not as a full-network one. The
    drift and autocorrelation statistics use full-parameter gradients.
    """
    p = torch.softmax(logits.float(), dim=1)
    y = torch.zeros_like(p).scatter_(1, targets[:, None], 1.0)
    d = p - y
    n = d.shape[0]
    hf = h.float()
    sq = (d * d).sum(1) * ((hf * hf).sum(1) + 1.0)
    meanW = d.t().mm(hf) / n
    meanb = d.mean(0)
    mean_norm_sq = float((meanW * meanW).sum() + (meanb * meanb).sum())
    return mean_norm_sq / (float(sq.mean()) + 1e-30)


def class_entropy(labels, n_classes):
    c = np.bincount(labels, minlength=n_classes).astype(float)
    p = c / max(c.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def evaluate_scheme(model, dataset, batches, g_ref, criterion, device, scores,
                    targets, n_params, max_lag, bn_mode):
    """Evaluate one batch scheme at frozen theta. The model is never updated.

    `dataset` must be the UNAUGMENTED dataset, and the same one used for g_ref,
    so eps_t reflects batch composition alone.
    """
    tap = FeatureTap(model)
    snap = _snapshot_bn(model)
    if bn_mode == "train":
        model.train()
    else:
        model.eval()

    eps_sum = torch.zeros(n_params, device=device)
    mom = torch.zeros(n_params, device=device)     # r_t = mu*r_{t-1} + eps_t
    g_t = torch.zeros(n_params, device=device)
    ring = []
    lag_num = [0.0] * (max_lag + 1)
    lag_cnt = [0] * (max_lag + 1)

    cos_ref, eps_norm, drift, mom_norm = [], [], [], []
    coh_w, diff_var, cls_ent = [], [], []
    g_ref_norm = float(g_ref.norm())

    for idx in batches:
        xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
        ys = torch.tensor([int(targets[i]) for i in idx]).to(device)

        model.zero_grad(set_to_none=False)
        logits = model(xs)
        criterion(logits, ys).backward()
        _flat_grad(model, g_t)

        eps = g_t - g_ref
        eps_sum.add_(eps)
        mom.mul_(MOMENTUM).add_(eps)

        gn = float(g_t.norm())
        cos_ref.append(float(g_t.dot(g_ref)) / (gn * g_ref_norm + 1e-30))
        eps_norm.append(float(eps.norm()))
        drift.append(float(eps_sum.norm()))
        mom_norm.append(float(mom.norm()))

        e = eps / (eps.norm() + 1e-30)
        for lag in range(1, min(max_lag, len(ring)) + 1):
            lag_num[lag] += float(e.dot(ring[-lag]))
            lag_cnt[lag] += 1
        ring.append(e.clone())
        if len(ring) > max_lag:
            ring.pop(0)

        coh_w.append(per_example_classifier_coherence(logits, ys, tap.h))
        diff_var.append(float(np.var(scores[idx])))
        cls_ent.append(class_entropy(np.asarray([int(targets[i]) for i in idx]),
                                     NUM_CLASSES))

    model.zero_grad(set_to_none=True)
    _restore_bn(model, snap)
    tap.close()

    m = len(eps_norm)
    mean_eps = float(np.mean(eps_norm))
    # Normalised prefix drift: 1.0 = random walk, growth = coherent drift.
    dn = np.array([drift[i] / (math.sqrt(i + 1) * mean_eps + 1e-30)
                   for i in range(m)])
    mn = np.array(mom_norm) / (mean_eps + 1e-30)

    return {
        "n_batches": m,
        "mean_cos_ref": float(np.mean(cos_ref)),
        "mean_eps_norm": mean_eps,
        "grad_ref_norm": g_ref_norm,
        "autocorr": {str(l): (lag_num[l] / lag_cnt[l] if lag_cnt[l] else None)
                     for l in range(1, max_lag + 1)},
        # order-SENSITIVE
        "drift_norm_mean": float(dn.mean()),
        "drift_norm_max": float(dn.max()),
        "drift_norm_auc": float(dn.sum() / m),
        "mom_norm_mean": float(mn.mean()),
        "mom_norm_max": float(mn.max()),
        "mom_norm_auc": float(mn.sum() / m),
        # order-INVARIANT; correctness check only
        "drift_final": drift[-1],
        "drift_norm_final": float(dn[-1]),
        # composition descriptors
        "within_batch_clf_coherence": float(np.mean(coh_w)),
        "within_batch_difficulty_var": float(np.mean(diff_var)),
        "within_batch_class_entropy": float(np.mean(cls_ent)),
        "drift_trajectory": [float(x) for x in dn[::max(1, m // 50)]],
    }


def reference_gradient(model, dataset, universe, criterion, device, n_params,
                       bn_mode, batch_size):
    """Reference gradient, computed in the SAME BatchNorm mode the schemes use.

    This matters more than it looks. If the reference is taken with BN in eval
    mode (running statistics) while the schemes run with BN in train mode
    (per-batch statistics), every eps_t inherits the same BN-mode offset. That
    common-mode component shows up as near-zero cos_ref, lag-1 autocorrelation
    around +0.8, and linear drift -- IN EVERY SCHEME, including shuffled and
    stride. It looks exactly like a curriculum effect and is not one.

    bn_mode 'eval': BN uses running statistics, so the full-data gradient over
        the universe is well defined. Accumulate it exactly, in sample-weighted
        form, at a large batch size.

    bn_mode 'train': BN uses per-batch statistics, so there is no full-data
        gradient -- the quantity depends on how the data is batched. The
        correct reference is then the EXPECTED MINIBATCH GRADIENT UNDER RANDOM
        BATCHING at the same batch size the schemes use, estimated by averaging
        over random partitions. Batch size must match, or the BN statistics
        differ systematically and the offset returns.

    BN running statistics are snapshotted and restored either way.
    """
    snap = _snapshot_bn(model)
    g = torch.zeros(n_params, device=device)
    g_t = torch.zeros(n_params, device=device)

    if bn_mode == "eval":
        model.eval()
        bs, total = 256, 0
        for s in range(0, len(universe), bs):
            idx = universe[s:s + bs]
            xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
            ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
            model.zero_grad(set_to_none=False)
            criterion(model(xs), ys).backward()
            _flat_grad(model, g_t)
            g.add_(g_t * len(idx))
            total += len(idx)
        g.div_(total)
    else:
        model.train()
        rng = np.random.default_rng(0)      # fixed: reference must not vary
        nb = len(universe) // batch_size
        n_used = 0
        for _rep in range(2):               # two independent random partitions
            perm = universe[rng.permutation(len(universe))]
            for t in range(nb):
                idx = perm[t * batch_size:(t + 1) * batch_size]
                xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
                ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
                model.zero_grad(set_to_none=False)
                criterion(model(xs), ys).backward()
                _flat_grad(model, g_t)
                g.add_(g_t)
                n_used += 1
        g.div_(n_used)

    model.zero_grad(set_to_none=True)
    _restore_bn(model, snap)
    return g


def score_samples(model, dataset, device, num_workers):
    """Per-sample loss at the current theta, unaugmented. Higher = harder."""
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    out = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out.append(F.cross_entropy(model(x), y, reduction="none").cpu())
    return torch.cat(out).numpy().astype(np.float32)


def validate_proxy(model, dataset, batches, criterion, device, n_params,
                   n_check=20):
    clf_params = list(model._fc_.parameters())
    clf_n = sum(p.numel() for p in clf_params)
    full, clf = [], []
    g_t = torch.zeros(n_params, device=device)
    for idx in batches[:n_check]:
        xs = torch.stack([dataset[int(i)][0] for i in idx]).to(device)
        ys = torch.tensor([dataset[int(i)][1] for i in idx]).to(device)
        model.zero_grad(set_to_none=False)
        criterion(model(xs), ys).backward()
        _flat_grad(model, g_t)
        full.append(g_t.clone())
        clf.append(torch.cat([p.grad.detach().reshape(-1)
                              for p in clf_params]).clone())
    model.zero_grad(set_to_none=True)
    fc, cc = [], []
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            fc.append(float(F.cosine_similarity(full[i], full[j], dim=0)))
            cc.append(float(F.cosine_similarity(clf[i], clf[j], dim=0)))
    fc, cc = np.array(fc), np.array(cc)
    r = float(np.corrcoef(fc, cc)[0, 1]) if len(fc) > 2 else float("nan")
    return {"n_pairs": len(fc), "pearson_full_vs_clf_cosines": r,
            "mean_full_cos": float(fc.mean()), "mean_clf_cos": float(cc.mean()),
            "clf_param_fraction": clf_n / n_params,
            "scope_note": "validates BATCH-gradient cosine correspondence "
                          "only, not per-example coherence"}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def checkpoint_analysis(model, ds_plain, criterion, device, n_params, rng, nw,
                        args, external_scores):
    scores = (external_scores if external_scores is not None
              else score_samples(model, ds_plain, device, nw))
    targets = np.asarray(ds_plain.targets)

    universe = choose_universe(len(ds_plain), BATCH_SIZE, args.max_batches, rng)
    g_ref = reference_gradient(model, ds_plain, universe, criterion, device,
                               n_params, args.bn_mode, BATCH_SIZE)

    out = {"schemes": {}, "universe_size": int(len(universe)),
           "bn_mode": args.bn_mode}
    for scheme in SCHEMES:
        batches = build_batches(scheme, universe, scores, BATCH_SIZE, rng)
        r = evaluate_scheme(model, ds_plain, batches, g_ref, criterion, device,
                            scores, targets, n_params, MAX_LAG, args.bn_mode)
        out["schemes"][scheme] = r
        print(f"    {scheme:<15} cos_ref={r['mean_cos_ref']:+.4f} "
              f"driftN_max={r['drift_norm_max']:.3f} "
              f"momN_max={r['mom_norm_max']:.3f} "
              f"lag1={r['autocorr']['1']:+.4f} "
              f"wb_coh={r['within_batch_clf_coherence']:.4f}")

    # Correctness check: identical contents => identical final drift.
    a = out["schemes"]["holistic_asc"]["drift_final"]
    b = out["schemes"]["holistic_bshuf"]["drift_final"]
    rel = abs(a - b) / (abs(a) + 1e-30)
    out["permutation_check"] = {"asc_drift_final": a, "bshuf_drift_final": b,
                                "rel_diff": rel, "ok": bool(rel < 1e-3)}
    print(f"    permutation check: |asc-bshuf|/asc = {rel:.2e} "
          f"{'OK' if rel < 1e-3 else 'FAILED -- universes differ'}")

    if args.validate_proxy:
        b0 = build_batches("shuffled", universe, scores, BATCH_SIZE, rng)
        out["proxy_validation"] = validate_proxy(model, ds_plain, b0, criterion,
                                                 device, n_params)
        print(f"    proxy pearson = "
              f"{out['proxy_validation']['pearson_full_vs_clf_cosines']:+.3f}")
    return out


def load_static_scores(path, metric, n_expected):
    """Load one metric column from a Gradus metric-scores parquet.

    Expected schema: an int64 `index` column giving the dataset sample index,
    a `class` column, and one float64 column per metric (color-variance,
    compression-ratio, edge-density, spatial-frequency, wavelet-energy,
    wavelet-entropy, convergence-time, saturation-time).

    Rows are sorted by `index` so the returned array is in dataset index order
    regardless of how the file was written.

    Orientation does not need to be known. The diagnostic evaluates
    holistic_asc and holistic_desc, so both directions of the ranking are
    measured either way.

    Note on seeds: the image-statistic metrics should be seed-invariant, but
    convergence-time and saturation-time are model-derived. Match the parquet
    seed to --seed when using those.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if metric is None:
        cols = [c for c in df.columns if c not in ("index", "class")]
        raise SystemExit("--static-metric is required. Available in "
                         f"{path}: {', '.join(cols)}")
    if metric not in df.columns:
        cols = [c for c in df.columns if c not in ("index", "class")]
        raise SystemExit(f"metric '{metric}' not in {path}. "
                         f"Available: {', '.join(cols)}")
    if "index" in df.columns:
        df = df.sort_values("index")
        idx = df["index"].to_numpy()
        if idx[0] != 0 or idx[-1] != len(idx) - 1:
            raise SystemExit(f"`index` column is not a contiguous 0..N-1 range "
                             f"(min {idx.min()}, max {idx.max()}, n {len(idx)}); "
                             f"cannot align to dataset order")
    v = df[metric].to_numpy().astype(np.float32)
    if len(v) != n_expected:
        raise SystemExit(f"{metric}: {len(v)} rows != {n_expected} samples")
    if not np.isfinite(v).all():
        bad = int((~np.isfinite(v)).sum())
        raise SystemExit(f"{metric}: {bad} non-finite values")
    print(f"  static difficulty: {metric} from {os.path.basename(path)} "
          f"[min {v.min():.4g}, max {v.max():.4g}, "
          f"n_unique {len(np.unique(v))}]")
    return v


def run(args):
    _lazy_imports()
    if os.path.exists(args.output_path):
        raise SystemExit(f"{args.output_path} exists. Use a distinct path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    nw = get_system_core_count()

    ds_aug, ds_plain, ds_test = build_datasets()
    train_loader = DataLoader(ds_aug, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False,
                             num_workers=nw, pin_memory=True)

    external_scores = None
    if args.static_scores:
        external_scores = load_static_scores(args.static_scores,
                                             args.static_metric,
                                             len(ds_plain))

    model = ResNet18(input_shape=(3, 32, 32), num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(args.seed + 777)

    cps = tuple(int(c) for c in args.checkpoints.split(","))
    log = {"seed": args.seed, "epochs": args.epochs, "batch_size": BATCH_SIZE,
           "schemes": list(SCHEMES), "checkpoints": list(cps),
           "max_batches": args.max_batches, "max_lag": MAX_LAG,
           "momentum": MOMENTUM, "n_params": n_params,
           "bn_mode": args.bn_mode,
           "augmentation": "none (g_ref and all schemes use the unaugmented "
                           "dataset)",
           "difficulty_source": (f"{args.static_metric} from "
                                 f"{os.path.basename(args.static_scores)}"
                                 if args.static_scores
                                 else "current per-sample loss at checkpoint"),
           "note": "frozen theta; no scheme updates the model",
           "checkpoint_analyses": {}, "train_log": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        rl, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            rl += loss.item() * y.size(0)
            seen += y.size(0)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        log["train_log"].append({"epoch": epoch,
                                 "lr": opt.param_groups[0]["lr"],
                                 "train_loss": rl / seen, "test_acc": acc})
        print(f"ep {epoch:3d} loss={rl/seen:.4f} test={acc:.2f}")

        if epoch in cps:
            print(f"  frozen-theta analysis at epoch {epoch} "
                  f"(BN mode: {args.bn_mode})")
            log["checkpoint_analyses"][str(epoch)] = checkpoint_analysis(
                model, ds_plain, criterion, device, n_params, rng, nw, args,
                external_scores)

        sched.step()

    d = os.path.dirname(args.output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n-> {args.output_path}")


def analyze(path):
    log = json.load(open(path))
    print(f"file: {path}  seed={log['seed']}  "
          f"BN={log.get('bn_mode')}  aug={log.get('augmentation')}")
    print(f"difficulty: {log.get('difficulty_source')}\n")

    for ep in log["checkpoints"]:
        key = str(ep)
        if key not in log["checkpoint_analyses"]:
            continue
        CA = log["checkpoint_analyses"][key]
        S = CA["schemes"]
        print(f"=== epoch {ep}  (universe {CA['universe_size']} samples) ===")
        print(f"  {'scheme':<16}{'cos_ref':>9}{'|eps|':>9}{'lag1':>8}{'lag5':>8}"
              f"{'driftN_mn':>11}{'driftN_mx':>11}{'momN_mx':>9}"
              f"{'wb_coh':>9}{'clsH':>7}")
        for s in log["schemes"]:
            if s not in S:
                continue
            r = S[s]
            l5 = r["autocorr"].get("5")
            print(f"  {s:<16}{r['mean_cos_ref']:>+9.4f}{r['mean_eps_norm']:>9.4f}"
                  f"{r['autocorr']['1']:>+8.4f}"
                  f"{(l5 if l5 is not None else float('nan')):>+8.4f}"
                  f"{r['drift_norm_mean']:>11.3f}{r['drift_norm_max']:>11.3f}"
                  f"{r['mom_norm_max']:>9.3f}"
                  f"{r['within_batch_clf_coherence']:>9.4f}"
                  f"{r['within_batch_class_entropy']:>7.3f}")

        sh = S.get("shuffled")
        if sh is not None and (sh["autocorr"]["1"] > 0.2
                               or sh["drift_norm_max"] > 3.0):
            print(f"\n  WARNING: shuffled itself shows lag1="
                  f"{sh['autocorr']['1']:+.3f} and driftN_max="
                  f"{sh['drift_norm_max']:.1f}. Random batching cannot produce "
                  f"correlated error. This is a COMMON-MODE OFFSET between the "
                  f"reference gradient and the scheme gradients (usually a BN "
                  f"mode or batch-size mismatch), not a curriculum effect. "
                  f"Do not interpret this checkpoint.")

        pc = CA.get("permutation_check", {})
        if pc:
            print(f"\n  permutation check (identical contents => identical "
                  f"final drift): rel diff {pc['rel_diff']:.2e} "
                  f"{'OK' if pc['ok'] else 'FAILED -- comparison invalid'}")

        try:
            sh, bs_, asc = S["shuffled"], S["holistic_bshuf"], S["holistic_asc"]
            print("\n  (A) within-batch homogeneity: shuffled vs holistic_bshuf "
                  "(order random in both)")
            print(f"      |eps|      {sh['mean_eps_norm']:.4f} -> "
                  f"{bs_['mean_eps_norm']:.4f}"
                  f"   ({bs_['mean_eps_norm']/sh['mean_eps_norm']:.2f}x)")
            print(f"      wb_coh     {sh['within_batch_clf_coherence']:.4f} -> "
                  f"{bs_['within_batch_clf_coherence']:.4f}")
            print(f"      momN_max   {sh['mom_norm_max']:.3f} -> "
                  f"{bs_['mom_norm_max']:.3f}")
            print("\n  (B) temporal ordering: holistic_bshuf vs holistic_asc "
                  "(identical batch vectors, different order)")
            print(f"      lag1       {bs_['autocorr']['1']:+.4f} -> "
                  f"{asc['autocorr']['1']:+.4f}")
            print(f"      driftN_max {bs_['drift_norm_max']:.3f} -> "
                  f"{asc['drift_norm_max']:.3f}")
            print(f"      momN_max   {bs_['mom_norm_max']:.3f} -> "
                  f"{asc['mom_norm_max']:.3f}"
                  f"   ({asc['mom_norm_max']/bs_['mom_norm_max']:.2f}x)")
            print(f"      momN_auc   {bs_['mom_norm_auc']:.3f} -> "
                  f"{asc['mom_norm_auc']:.3f}")
        except KeyError:
            pass

        pv = CA.get("proxy_validation")
        if pv:
            print(f"\n  classifier proxy pearson = "
                  f"{pv['pearson_full_vs_clf_cosines']:+.3f} "
                  f"({100*pv['clf_param_fraction']:.2f}% of params); "
                  f"{pv['scope_note']}")
        print()

    print("Read (A) and (B) as effect sizes, not verdicts: neither comparison "
          "has a significance test, and this is one seed at one theta.")
    print("These statistics are computed by this script ad hoc; rerun them in "
          "your own analysis code before citing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--output-path", type=str)
    ap.add_argument("--checkpoints", type=str,
                    default=",".join(str(c) for c in CHECKPOINTS))
    ap.add_argument("--max-batches", type=int, default=MAX_BATCHES,
                    help="batches per scheme; the sample universe is "
                         "subsampled once so every scheme covers identical "
                         "samples")
    ap.add_argument("--bn-mode", choices=["eval", "train"], default="eval",
                    help="eval freezes BN running stats and isolates pure "
                         "gradient geometry; train lets homogeneous batches "
                         "form their own BN statistics (a separate candidate "
                         "mechanism). Running stats are restored either way.")
    ap.add_argument("--static-scores", type=str, default=None,
                    help="path to a Gradus metric-scores parquet. Without it "
                         "the difficulty is current per-sample loss, which "
                         "does NOT reproduce the original curricula.")
    ap.add_argument("--static-metric", type=str, default=None,
                    help="column to use from --static-scores, e.g. "
                         "saturation-time, edge-density, wavelet-entropy")
    ap.add_argument("--validate-proxy", action="store_true", default=True)
    ap.add_argument("--no-validate-proxy", dest="validate_proxy",
                    action="store_false")
    ap.add_argument("--analyze", type=str, default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    if args.seed is None or args.output_path is None:
        ap.error("--seed and --output-path are required")
    run(args)


if __name__ == "__main__":
    main()