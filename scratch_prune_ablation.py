"""# Sample-Level Pruning Ablation Harness

Trains <network> on <dataset> while excluding a scheduled fraction of the training set each epoch,
chosen by a configurable policy. Every policy with the same schedule processes EXACTLY the same
number of samples (quotas are enforced per epoch, per class), so DSI is matched by construction and
accuracy is the only quantity compared. Random pruning on the same schedule is the control.

## Policy axes
    --score         random | el2n | loss | aum | act_cv | static:<metric>
    --refresh       once   (score frozen at onset: mean over epochs 1..onset)
                    ema    (exponential moving average, updated whenever a sample is trained on)
    --drop          low | high     which end of the score is excluded
                        el2n/loss:  low = easy,  high = hard
                        aum:        low = hard,  high = easy
                        act_cv:     low = stable (sample-level B-PAS), high = unstable
    --stratify      none | class   enforce the pruning fraction within each class
    --mode          permanent      pruned samples never return (pruned set only grows)
                    temporary      the excluded set is re-chosen every epoch from all samples
    --temperature   0              deterministic: exclude exactly the top-quota by score
                    T > 0          stochastic: Gumbel-top-k on (standardized score / T); T -> inf
                                   recovers random. One knob interpolating score-ranked <-> random.
    --max-gap       K > 0          coverage guarantee (temporary mode): no sample may be excluded
                                   for more than K consecutive epochs. 0 = off.
    --tilt-start    E > 0          use the score only from epoch E on; exclusion is random before E.
    --label-noise   p > 0          flip a fraction p of TRAINING labels (fixed per seed; test labels
                                   stay clean). Mechanism probe.
    --noise-type    symmetric      flipped labels go to a uniformly random other class
                    pairflip       flipped labels go to the next class, (c + 1) mod C (Han et al., 2018)
    --objective     ce | gce       training loss; gce = Generalized Cross-Entropy, q = 0.7
                                   (Zhang & Sabuncu, 2018), a standard robust-loss baseline.
    --budget        fixed          exclude --prune-fraction per epoch (after onset/ramp)
                    adaptive       exclude clip(k * noise_estimate, 0, --prune-fraction) per epoch, where
                                   noise_estimate = fraction of samples with negative EMA margin
                                   (misclassified under their given label; cf. AUM, Pleiss et al., 2020)
    --budget-k      k              multiplier for the adaptive budget (default 2)
                    rule           THE METHOD. At the end of --estimate-epoch, read the noise rate p_hat
                                   (see below) and fix the exclusion budget and selection for the rest of
                                   training:  budget = clip(p_hat + 0.3, 0.3, 0.8);
                                   T = 1 on the configured score if p_hat >= 0.1, else random exclusion.
                                   --prune-fraction and --temperature are ignored.
    --rule-noise    p >= 0         (rule) use p instead of the estimate: oracle / misspecification probe.
                    -1             (default) estimate p from the trusted split (requires --trusted-size).
    --estimate-epoch E             (rule, estimated) epoch at whose end p_hat is read (default 8; <= onset).
    --trusted-size  N > 0          carve N training samples (class-stratified, fixed per seed) into a
                                   TRUSTED split: clean labels, never trained on, evaluated each epoch
                                   without augmentation. Every arm of a comparison must use the same N,
                                   so all arms train on the same samples.

## Anchored noise-rate estimate (logged whenever --trusted-size > 0)
    p_hat = (m - e) / (1 - e), clipped to [0, 1], where m = fraction of training samples with negative
    EMA margin and e = error on the trusted split. Follows from m ~ p + (1 - p) * e_clean.

## Logged diagnostics (every run)
    est_negmargin   fraction of samples with negative EMA margin (the adaptive-budget signal)
    est_el2n_gt1    fraction of samples with EMA EL2N > 1 ("confidently wrong": mass on another class)
    est_gmm         weight of the high-loss component of a 2-component GMM on EMA loss (DivideMix-style; plain EM)
    auc_*           (noisy runs) AUC of each EMA score for identifying label-flipped samples

## Calibration mode
    --stop-after K  train under the normal --epochs schedule but stop after epoch K, writing
                    calibration/<run_id>_stop<K>.json (per-epoch estimates & AUCs) instead of a
                    results row. Used to calibrate noise-rate estimators cheaply.
    --snapshot-epochs  (calibration mode) also save per-sample state at these epochs to
                    calibration/<run_id>_ep<E>.npz: EMA loss / EL2N / margin, last predicted class &
                    confidence, given (possibly noisy) label, true label, flipped mask. Lets noise-rate
                    estimators be designed & validated offline on CPU.

## Schedule
    Epochs 1..onset: full data.  Then the pruned fraction ramps linearly to --prune-fraction over
    --ramp epochs and holds; the final --full-final epochs return to full data (annealing).

## Usage (from the gradus repo root)
    python scratch_prune_ablation.py --score el2n --refresh ema --drop low --stratify class --mode temporary --seed 1
    python scratch_prune_ablation.py --summarize          # aggregate every recorded run -> summary.txt

## Outputs (under --output/<dataset>/<network>/)
    runs/<run_id>.json      Config, per-epoch curves, DSI, final/best accuracy.
    results.csv             One row per completed run (runs already recorded are skipped).
    summary.txt             Seed-averaged table with deltas vs. the matched random control.
"""

from argparse   import ArgumentParser, Namespace
from hashlib    import md5
from json       import dump, dumps
from pathlib    import Path
from time       import time
from typing     import Dict, List, Optional
import warnings

import numpy    as np
import pandas   as pd

POLICY_KEYS:    List[str] = [
                                "network", "dataset", "epochs", "batch_size", "score", "refresh",
                                "drop", "stratify", "mode", "prune_fraction", "onset", "ramp",
                                "full_final", "alpha", "window"
                            ]

# Keys added after stage 1. They enter the run hash only when non-default, so stage-1 runs keep
# their IDs (and are skipped, not re-run); older result rows are back-filled with these defaults.
EXTRA_DEFAULTS: Dict = {"temperature": 0.0, "max_gap": 0, "tilt_start": 0, "label_noise": 0.0,
                        "noise_type": "symmetric", "objective": "ce", "budget": "fixed", "budget_k": 0.0}

# Keys added in stage 10 (same hashing convention: only non-default values enter the run hash).
EXTRA_DEFAULTS |= {"trusted_size": 0, "rule_noise": -1.0, "estimate_epoch": 0}

GCE_Q:          float = 0.7

# The noise-rate rule (fixed from CIFAR-100 stages 5-6; validated with oracle p in stage 9b).
RULE_OFFSET:    float = 0.3
RULE_MIN:       float = 0.3
RULE_MAX:       float = 0.8
RULE_TILT_P:    float = 0.1

# ARGUMENTS ========================================================================================

def parse_arguments() -> Namespace:
    """# Parse Harness Arguments."""
    p = ArgumentParser(description = "Sample-level pruning ablation harness.")
    p.add_argument("--network",         type = str,     default = "resnet-18")
    p.add_argument("--dataset",         type = str,     default = "cifar-100")
    p.add_argument("--epochs",          type = int,     default = 100)
    p.add_argument("--seed",            type = int,     default = 1)
    p.add_argument("--batch-size",      type = int,     default = 128)
    p.add_argument("--workers",         type = int,     default = 4)
    p.add_argument("--root",            type = str,     default = ".cache/data")
    p.add_argument("--device",          type = str,     default = "auto", choices = ["auto", "cpu", "cuda"])
    p.add_argument("--scores",          type = str,     default = None,
                   help = "Static scores parquet (only for --score static:<metric>).")
    p.add_argument("--output",          type = str,     default = "analysis/pruning")

    p.add_argument("--score",           type = str,     default = "random")
    p.add_argument("--refresh",         type = str,     default = "ema",        choices = ["once", "ema"])
    p.add_argument("--drop",            type = str,     default = "low",        choices = ["low", "high"])
    p.add_argument("--stratify",        type = str,     default = "class",      choices = ["none", "class"])
    p.add_argument("--mode",            type = str,     default = "temporary",  choices = ["permanent", "temporary"])
    p.add_argument("--prune-fraction",  type = float,   default = 0.5)
    p.add_argument("--onset",           type = int,     default = 10)
    p.add_argument("--ramp",            type = int,     default = 10)
    p.add_argument("--full-final",      type = int,     default = 0)
    p.add_argument("--alpha",           type = float,   default = 0.5,  help = "EMA weight on the newest observation.")
    p.add_argument("--window",          type = int,     default = 5,    help = "Epoch window for act_cv.")
    p.add_argument("--temperature",     type = float,   default = 0.0,
                   help = "0 = deterministic top-k; > 0 = Gumbel-top-k sampling on standardized scores.")
    p.add_argument("--max-gap",         type = int,     default = 0,
                   help = "Max consecutive epochs a sample may be excluded (temporary mode). 0 = off.")
    p.add_argument("--tilt-start",      type = int,     default = 0,
                   help = "Epoch from which the score is used; exclusion is random before it. 0 = from onset.")
    p.add_argument("--label-noise",     type = float,   default = 0.0,
                   help = "Fraction of training labels flipped (fixed per seed).")
    p.add_argument("--noise-type",      type = str,     default = "symmetric",  choices = ["symmetric", "pairflip"])
    p.add_argument("--objective",       type = str,     default = "ce",         choices = ["ce", "gce"])
    p.add_argument("--budget",          type = str,     default = "fixed",      choices = ["fixed", "adaptive", "rule"])
    p.add_argument("--budget-k",        type = float,   default = 2.0,
                   help = "Adaptive budget = clip(k * noise_estimate, 0, prune_fraction).")
    p.add_argument("--rule-noise",      type = float,   default = -1.0,
                   help = "(--budget rule) Noise rate fed to the rule; -1 = estimate it from the trusted split.")
    p.add_argument("--estimate-epoch",  type = int,     default = 8,
                   help = "(--budget rule, estimated) Epoch at whose end the noise rate is read.")
    p.add_argument("--trusted-size",    type = int,     default = 0,
                   help = "Carve this many clean-labeled training samples into a held-out trusted split.")

    p.add_argument("--stop-after",      type = int,     default = 0,
                   help = "Calibration mode: stop after this epoch and write a calibration record (0 = off).")
    p.add_argument("--snapshot-epochs", type = str,     default = "5,10,15",
                   help = "Calibration mode: comma-separated epochs at which per-sample state is saved.")
    p.add_argument("--summarize",       action = "store_true", help = "Aggregate recorded runs and exit.")
    args = p.parse_args()

    # Normalize policy fields that are irrelevant to the chosen score (so configs hash identically).
    if args.score == "random" or args.score.startswith("static:"):  args.refresh, args.alpha = "n/a", 0.0
    if args.score != "act_cv":                                      args.window = 0
    if args.score == "act_cv":                                      args.refresh = "window"
    if args.score == "random":                                      args.drop, args.temperature, args.tilt_start = "n/a", 0.0, 0
    if args.mode == "permanent":                                    args.max_gap = 0
    if args.prune_fraction == 0:
        args.score, args.refresh, args.drop, args.stratify, args.mode = "none", "n/a", "n/a", "none", "n/a"
        args.onset = args.ramp = args.full_final = 0;   args.alpha = 0.0;   args.window = 0
        args.temperature, args.max_gap, args.tilt_start = 0.0, 0, 0
    if args.label_noise == 0:                                       args.noise_type = "symmetric"
    if args.budget == "fixed" or args.prune_fraction == 0:          args.budget, args.budget_k = "fixed", 0.0
    if args.budget == "rule":
        # The rule chooses budget & temperature at run time; the CLI values are placeholders.
        args.prune_fraction, args.temperature, args.budget_k = RULE_MAX, 0.0, 0.0
        if args.score == "random": p.error("--budget rule needs a score for the tilted branch (e.g. --score el2n --drop high)")
        if args.rule_noise < 0:
            if args.trusted_size <= 0:          p.error("--budget rule without --rule-noise needs --trusted-size > 0")
            if args.estimate_epoch > args.onset: p.error("--estimate-epoch must be <= --onset")
    if args.budget != "rule":                                       args.rule_noise = -1.0
    if args.budget != "rule" or args.rule_noise >= 0:               args.estimate_epoch = 0
    return args

# SCHEDULE & SELECTION (pure numpy) ================================================================

def target_fraction(epoch: int, args: Namespace, noise_estimate: Optional[float] = None,
                    rule_level: Optional[float] = None) -> float:
    """# Fraction of the training set excluded during `epoch` (1-indexed).

    fixed:      --prune-fraction.
    adaptive:   clip(k * noise_estimate, 0, --prune-fraction), noise_estimate from the previous epoch.
    rule:       the level fixed by the noise-rate rule (0 until it has been decided).
    All are scaled by the linear ramp after onset.
    """
    if args.prune_fraction == 0 or epoch <= args.onset or epoch > args.epochs - args.full_final: return 0.0
    if args.budget == "rule":   level = rule_level or 0.0
    elif args.budget == "fixed": level = args.prune_fraction
    else:                       level = float(np.clip(args.budget_k * (noise_estimate or 0.0), 0.0, args.prune_fraction))
    return level * min(1.0, (epoch - args.onset) / max(1, args.ramp))

def noise_rule(p: float):
    """# The method's rule: noise rate -> (exclusion budget, temperature); T = 0 means random exclusion."""
    p =         float(np.clip(p, 0.0, 1.0))
    return float(np.clip(p + RULE_OFFSET, RULE_MIN, RULE_MAX)), (1.0 if p >= RULE_TILT_P else 0.0)

def anchored_estimate(negmargin: float, trusted_err: float) -> float:
    """# p_hat = (m - e) / (1 - e), clipped to [0, 1]: training negative-margin fraction corrected by the
    error on clean held-out (trusted) data at the same epoch."""
    if not np.isfinite(negmargin) or not np.isfinite(trusted_err) or trusted_err >= 1: return float("nan")
    return float(np.clip((negmargin - trusted_err) / (1 - trusted_err), 0.0, 1.0))

def carve_trusted(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """# Class-stratified boolean mask of `size` samples held out as the trusted split (fixed per seed)."""
    mask =      np.zeros(len(labels), dtype = bool)
    if size <= 0: return mask
    rng =       np.random.default_rng(20_000 + seed)
    classes =   np.unique(labels)
    per_class = np.full(len(classes), size // len(classes));    per_class[: size % len(classes)] += 1
    for c, k in zip(classes, per_class):
        mask[rng.choice(np.flatnonzero(labels == c), size = k, replace = False)] = True
    return mask

def estimate_noise(state) -> Dict[str, float]:
    """# Online noise-rate estimates from EMA training statistics.

    negmargin:  fraction of samples whose EMA margin (true logit - best other) is negative.
    gmm:        weight of the higher-mean component of a 2-component Gaussian mixture on EMA loss.
    """
    margin =    state.ema["aum"];   loss = state.ema["loss"];   el2n = state.ema["el2n"]
    seen =      np.isfinite(margin)
    out =       {"negmargin":   float((margin[seen] < 0).mean()) if seen.any() else float("nan"),
                 "el2n_gt1":    float((el2n[seen] > 1.0).mean()) if seen.any() else float("nan"),
                 "gmm":         float("nan")}
    x =         loss[np.isfinite(loss)]
    if len(x) > 100: out["gmm"] = gmm_high_weight(x)
    return out

def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """# ROC AUC of `score` for separating `positive` samples (Mann-Whitney; NaN scores dropped)."""
    keep =      np.isfinite(score)
    s, y =      score[keep], positive[keep]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    ranks =     pd.Series(s).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def gmm_high_weight(x: np.ndarray, iterations: int = 50) -> float:
    """# Weight of the higher-mean component of a 1-D, 2-component Gaussian mixture (plain EM)."""
    lo, hi =    np.quantile(x, [0.25, 0.9])
    mu =        np.array([lo, hi], dtype = np.float64)
    var =       np.full(2, x.var() + 1e-6)
    w =         np.array([0.5, 0.5])
    for _ in range(iterations):
        log_p = -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2 * np.pi * var)) + np.log(w)
        log_p -= log_p.max(axis = 1, keepdims = True)
        r =     np.exp(log_p);  r /= r.sum(axis = 1, keepdims = True)
        nk =    r.sum(axis = 0) + 1e-12
        w =     nk / len(x)
        mu =    (r * x[:, None]).sum(axis = 0) / nk
        var =   (r * (x[:, None] - mu) ** 2).sum(axis = 0) / nk + 1e-4
    return float(w[int(np.argmax(mu))])

def select_excluded(
    scores:         Optional[np.ndarray],
    excluded:       np.ndarray,
    fraction:       float,
    labels:         np.ndarray,
    stratify:       str,
    mode:           str,
    drop:           str,
    rng:            np.random.Generator,
    temperature:    float =                 0.0,
    forced:         Optional[np.ndarray] =  None
) -> np.ndarray:
    """# Boolean mask of samples excluded next epoch.

    Quotas are exact per group (class, or the whole set), so every policy with the same schedule
    excludes the same number of samples. Unscored (NaN) samples are excluded last. Samples flagged
    in `forced` (coverage guarantee) are ineligible for exclusion.

    Score-based exclusion ranks keys = z / T + Gumbel noise, where z is the within-group standardized
    score oriented so that larger z = more excludable. T = 0 is deterministic top-k; as T grows the
    selection approaches uniform random (Gumbel-top-k = sampling without replacement ~ exp(z / T)).
    """
    if fraction == 0: return np.zeros(len(labels), dtype = bool)

    groups =    labels if stratify == "class" else np.zeros(len(labels), dtype = np.int64)
    new =       excluded.copy() if mode == "permanent" else np.zeros(len(labels), dtype = bool)

    for g in np.unique(groups):
        members =   np.flatnonzero(groups == g)
        quota =     int(round(fraction * len(members)))
        if mode == "permanent":
            candidates =    members[~new[members]]
            needed =        quota - int(new[members].sum())
        else:
            candidates, needed = members, quota
            if forced is not None: candidates = candidates[~forced[candidates]]
        if needed <= 0 or len(candidates) == 0: continue
        needed =    min(needed, len(candidates))

        if scores is None:
            chosen =    rng.choice(candidates, size = needed, replace = False)
        else:
            s =         scores[candidates].astype(np.float64)
            z =         (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)
            z =         -z if drop == "low" else z                       # larger z = excluded first
            key =       z / temperature + rng.gumbel(size = len(z)) if temperature > 0 \
                        else z + rng.uniform(0, 1e-9, size = len(z))     # random tie-breaking
            key[np.isnan(key)] = -np.inf                                 # unscored excluded last
            chosen =    candidates[np.argsort(-key, kind = "stable")[:needed]]
        new[chosen] = True

    return new

def corrupt_labels(labels: np.ndarray, p: float, num_classes: int, seed: int, kind: str = "symmetric"):
    """# Label noise on a fraction p of samples (fixed per seed).

    symmetric:  flip to a uniformly random OTHER class.
    pairflip:   flip to the next class, (c + 1) mod C - structured, class-dependent noise.
    """
    noisy = np.zeros(len(labels), dtype = bool)
    if p <= 0: return labels, noisy
    rng =           np.random.default_rng(10_000 + seed)
    noisy =         rng.random(len(labels)) < p
    flipped =       labels.copy()
    offset =        rng.integers(1, num_classes, size = int(noisy.sum())) if kind == "symmetric" else 1
    flipped[noisy] = (labels[noisy] + offset) % num_classes
    return flipped, noisy

def training_loss(logits, targets, objective: str):
    """# Per-sample training loss: cross-entropy, or Generalized Cross-Entropy (1 - p_y^q) / q."""
    from torch.nn.functional import cross_entropy, softmax
    if objective == "ce": return cross_entropy(logits, targets, reduction = "none")
    p_y = softmax(logits, dim = 1).gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-7)
    return (1 - p_y ** GCE_Q) / GCE_Q

# SCORE STATE ======================================================================================

class ScoreState:
    """# Per-Sample Statistics Accumulated from Training Passes."""

    def __init__(self, n: int, epochs: int, args: Namespace, static: Optional[np.ndarray]):
        self.args =         args
        self.static =       static
        self.ema =          {k: np.full(n, np.nan) for k in ("el2n", "loss", "aum")}
        self.once_sum =     {k: np.zeros(n) for k in ("el2n", "loss", "aum")}
        self.once_count =   np.zeros(n)
        self.act =          np.full((n, epochs), np.nan, dtype = np.float32) if args.score == "act_cv" else None

    def update(self, idx: np.ndarray, epoch: int, values: Dict[str, np.ndarray], act: Optional[np.ndarray]) -> None:
        """# Fold one batch of observations into the state.

        Runs whose score ignores the EMA (random, static, full data) have alpha normalized to 0 for
        hashing; the EMA itself still uses 0.5 there so the logged noise estimates stay meaningful.
        """
        a = self.args.alpha if self.args.alpha > 0 else 0.5
        for k, v in values.items():
            prev =          self.ema[k][idx]
            self.ema[k][idx] = np.where(np.isnan(prev), v, a * v + (1 - a) * prev)
            if epoch <= self.args.onset: self.once_sum[k][idx] += v
        if epoch <= self.args.onset: self.once_count[idx] += 1
        if self.act is not None and act is not None: self.act[idx, epoch - 1] = act

    def scores(self, epoch: int) -> Optional[np.ndarray]:
        """# Current ranking score (None = random policy)."""
        score = self.args.score
        if score in ("random", "none"):     return None
        if score.startswith("static:"):     return self.static
        if score == "act_cv":
            # CV over each sample's last `window` OBSERVED epochs (excluded epochs are skipped, so
            # pruned samples keep a defined, if stale, score).
            obs =   self.act[:, :epoch]
            seen =  ~np.isnan(obs)
            recent = seen & (np.cumsum(seen[:, ::-1], axis = 1)[:, ::-1] <= self.args.window)
            w =     np.where(recent, obs, np.nan)
            with warnings.catch_warnings(), np.errstate(all = "ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanstd(w, axis = 1) / np.nanmean(w, axis = 1)
        if self.args.refresh == "once":
            with np.errstate(all = "ignore"):
                return self.once_sum[score] / self.once_count
        return self.ema[score]

# TRAINING =========================================================================================

def run(args: Namespace, run_id: str, run_dir: Path) -> Dict:
    """# Train One Configuration Under Its Pruning Policy."""
    from torch                      import Generator, no_grad, randperm, stack
    from torch.nn.functional        import cross_entropy, one_hot, softmax
    from torch.optim                import SGD
    from torch.optim.lr_scheduler   import CosineAnnealingLR
    from torch.utils.data           import DataLoader, Dataset, Sampler, Subset
    from tqdm                       import tqdm

    from gradus.registration        import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities           import determine_device, set_seed
    from copy                       import copy

    class Indexed(Dataset):
        """Yield (sample, target, index), with targets taken from a (possibly noisy) label array."""
        def __init__(self, data, targets):  self.data, self.targets = data, targets
        def __len__(self):                  return len(self.data)
        def __getitem__(self, i):           x, _ = self.data[i]; return x, int(self.targets[i]), i

    class ActiveSampler(Sampler):
        """Reshuffled pass over the currently active indices (changed between epochs)."""
        def __init__(self, indices, seed):  self.indices, self.gen = indices, Generator().manual_seed(seed)
        def __len__(self):                  return len(self.indices)
        def __iter__(self):
            order = randperm(len(self.indices), generator = self.gen).numpy()
            return iter(self.indices[order].tolist())

    set_seed(args.seed);    device = determine_device(args.device)
    rng =                   np.random.default_rng(args.seed)

    dataset =               DATASET_REGISTRY.load_dataset(
                                dataset_id = args.dataset, epochs = args.epochs, seed = args.seed,
                                root = args.root, batch_size = args.batch_size, max_workers = args.workers
                            )
    train =                 dataset.train_data
    N, E =                  len(train), args.epochs
    labels =                np.array(getattr(train, "targets", [train[i][1] for i in range(N)]), dtype = np.int64)
    true_labels =           labels.copy()
    labels, noisy =         corrupt_labels(labels, args.label_noise, dataset.num_classes, args.seed, args.noise_type)

    # Trusted split: clean labels, never trained on; evaluated without augmentation (test transform).
    trusted =               carve_trusted(true_labels, args.trusted_size, args.seed)
    labels[trusted] =       true_labels[trusted];   noisy[trusted] = False
    trainable =             np.flatnonzero(~trusted)
    trusted_loader =        None
    if trusted.any():
        view =              copy(train)
        if hasattr(view, "transform") and hasattr(dataset.test_data, "transform"): view.transform = dataset.test_data.transform
        trusted_loader =    DataLoader(Subset(view, np.flatnonzero(trusted).tolist()), batch_size = 512,
                                       shuffle = False, num_workers = args.workers)
    last_pred =             np.full(N, -1, dtype = np.int64)
    last_conf =             np.full(N, np.nan, dtype = np.float32)
    snapshots =             {int(e) for e in args.snapshot_epochs.split(",") if e.strip()} if args.stop_after else set()

    static =                None
    if args.score.startswith("static:"):
        path =              Path(args.scores or f".cache/scores/{args.dataset}/metric-scores_seed-1.parquet")
        static =            pd.read_parquet(path).set_index("index")[args.score.split(":", 1)[1]] \
                            .reindex(np.arange(N)).to_numpy(dtype = np.float64)

    sampler =               ActiveSampler(trainable.copy(), args.seed)
    loader =                DataLoader(Indexed(train, labels), batch_size = args.batch_size, sampler = sampler,
                                       num_workers = args.workers, pin_memory = True,
                                       persistent_workers = args.workers > 0)

    network =               NETWORK_REGISTRY.load_network(
                                network_id = args.network, input_shape = dataset.input_shape,
                                num_classes = dataset.num_classes
                            ).to(device)
    optimizer =             SGD(network.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    scheduler =             CosineAnnealingLR(optimizer, T_max = E)

    state =                 ScoreState(N, E, args, static)
    excluded =              np.zeros(N, dtype = bool)
    consecutive =           np.zeros(N, dtype = np.int64)
    estimate =              {"negmargin": float("nan"), "el2n_gt1": float("nan"), "gmm": float("nan")}
    history =               []
    processed =             0
    want_act =              args.score == "act_cv"
    rule =                  {"p": float("nan"), "rho": float("nan"), "T": float("nan"), "trusted_err": float("nan")}
    if args.budget == "rule" and args.rule_noise >= 0:
        rule["p"] =         args.rule_noise
        rule["rho"], rule["T"] = noise_rule(args.rule_noise)
    trusted_err =           float("nan")
    started =               time()

    for epoch in range(1, E + 1):

        # Choose this epoch's active set (decided from state through the previous epoch).
        # Selection runs over the trainable samples only (the trusted split is never trained on).
        fraction =          target_fraction(epoch, args, None if np.isnan(estimate["negmargin"]) else estimate["negmargin"],
                                            None if np.isnan(rule["rho"]) else rule["rho"])
        forced =            consecutive >= args.max_gap if args.max_gap > 0 else None
        use_score =         fraction > 0 and epoch > args.tilt_start and (args.budget != "rule" or rule["T"] > 0)
        temperature =       rule["T"] if args.budget == "rule" else args.temperature
        scores =            state.scores(epoch - 1) if use_score else None
        sub =               select_excluded(None if scores is None else scores[trainable], excluded[trainable],
                                            fraction, labels[trainable], args.stratify, args.mode, args.drop, rng,
                                            temperature, None if forced is None else forced[trainable])
        excluded =          np.zeros(N, dtype = bool);  excluded[trainable] = sub
        consecutive =       np.where(excluded, consecutive + 1, 0)
        sampler.indices =   trainable[~sub]
        processed +=        len(sampler.indices)

        network.train();    correct = seen = 0
        for x, y, idx in tqdm(loader, desc = f"[{run_id}] epoch {epoch}/{E}", leave = False):
            x, y =              x.to(device), y.to(device)
            out =               network(x, return_activations = True) if want_act else network(x)
            logits, acts =      out if want_act else (out, None)
            per_sample =        training_loss(logits, y, args.objective)
            optimizer.zero_grad();  per_sample.mean().backward();   optimizer.step()

            with no_grad():
                z =             logits.detach()
                onehot =        one_hot(y, num_classes = z.shape[1]).float()
                margin =        z.gather(1, y.unsqueeze(1)).squeeze(1) - \
                                z.masked_fill(onehot.bool(), float("-inf")).max(dim = 1).values
                values =        {
                                    "loss":     cross_entropy(z, y, reduction = "none").cpu().numpy(),
                                    "el2n":     (softmax(z, dim = 1) - onehot).norm(dim = 1).cpu().numpy(),
                                    "aum":      margin.cpu().numpy(),
                                }
                act =           stack([a.flatten(1).std(dim = 1) for a in acts]).mean(dim = 0).cpu().numpy() \
                                if want_act else None
                state.update(idx.numpy(), epoch, values, act)
                conf, pred =    softmax(z, dim = 1).max(dim = 1)
                last_pred[idx.numpy()] = pred.cpu().numpy();    last_conf[idx.numpy()] = conf.cpu().numpy()
                correct +=      z.argmax(dim = 1).eq(y).sum().item();  seen += y.size(0)

        scheduler.step()
        estimate =          estimate_noise(state)

        network.eval();     hits = total = 0
        with no_grad():
            for x, y in dataset.test_loader:
                hits +=     network(x.to(device)).argmax(dim = 1).eq(y.to(device)).sum().item()
                total +=    y.size(0)
            if trusted_loader is not None:    # targets here are the dataset's own (clean) labels
                wrong = count = 0
                for x, y in trusted_loader:
                    wrong +=    network(x.to(device)).argmax(dim = 1).ne(y.to(device)).sum().item()
                    count +=    y.size(0)
                trusted_err = wrong / count

        # The method: read the noise rate once, then fix budget & selection for the rest of training.
        anchored =          anchored_estimate(estimate["negmargin"], trusted_err)
        if args.budget == "rule" and args.rule_noise < 0 and epoch == args.estimate_epoch:
            rule["p"], rule["trusted_err"] = anchored, trusted_err
            rule["rho"], rule["T"] = noise_rule(0.0 if np.isnan(anchored) else anchored)
            print(f"[{run_id}] rule: p_hat = {anchored:.3f} (negmargin {estimate['negmargin']:.3f}, trusted err "
                  f"{trusted_err:.3f}) -> budget {rule['rho']:.2f}, " + (f"T = {rule['T']:g}" if rule["T"] > 0 else "random"))

        history.append({"epoch": epoch, "active": int(len(sampler.indices)),
                        "max_streak": int(consecutive.max()), "streak_ge_10": int((consecutive >= 10).sum()),
                        "excluded_frac": float(excluded[trainable].mean()),
                        "trusted_err": trusted_err, "est_anchored": anchored,
                        "est_negmargin": estimate["negmargin"], "est_el2n_gt1": estimate["el2n_gt1"],
                        "est_gmm": estimate["gmm"],
                        **({"auc_loss":     auc(state.ema["loss"], noisy),
                            "auc_el2n":     auc(state.ema["el2n"], noisy),
                            "auc_negmargin": auc(-state.ema["aum"], noisy)} if noisy.any() else {}),
                        "noisy_excluded_frac": float(excluded[noisy].mean()) if noisy.any() else None,
                        "train_acc": correct / max(1, seen), "val_acc": hits / total})
        print(f"[{run_id}] epoch {epoch:3}/{E}: active = {len(sampler.indices):5}/{len(trainable)}; "
              f"max streak = {int(consecutive.max()):3}; "
              + (f"noisy excluded = {excluded[noisy].mean():.3f}; " if noisy.any() else "") +
              f"excluded = {excluded[trainable].mean():.3f}; est noise = {estimate['negmargin']:.3f}; " +
              (f"p_hat = {anchored:.3f}; " if trusted_loader is not None else "") +
              f"train acc = {history[-1]['train_acc']:.4f}; val acc = {history[-1]['val_acc']:.4f}")

        # Calibration mode: per-sample snapshots for offline estimator design.
        if epoch in snapshots:
            run_dir.mkdir(parents = True, exist_ok = True)
            np.savez_compressed(run_dir / f"{run_id}_ep{epoch}.npz",
                                ema_loss = state.ema["loss"].astype(np.float32), ema_el2n = state.ema["el2n"].astype(np.float32),
                                ema_margin = state.ema["aum"].astype(np.float32), pred = last_pred, conf = last_conf,
                                given = labels, true = true_labels, noisy = noisy, epoch = epoch)

        # Calibration mode: stop early under the full schedule & write the trace instead of a result.
        if args.stop_after and epoch == args.stop_after:
            record = {"run_id": run_id, "seed": args.seed, "stop_after": epoch,
                      **{k: getattr(args, k) for k in POLICY_KEYS}, **{k: getattr(args, k) for k in EXTRA_DEFAULTS},
                      "history": history}
            run_dir.mkdir(parents = True, exist_ok = True)
            with open(run_dir / f"{run_id}_stop{epoch}.json", "w") as f: dump(record, f, indent = 2)
            return record

    val =                   [h["val_acc"] for h in history]
    result =                {
                                "run_id":       run_id,
                                "seed":         args.seed,
                                **{k: getattr(args, k) for k in POLICY_KEYS},
                                **{k: getattr(args, k) for k in EXTRA_DEFAULTS},
                                "final_acc":    val[-1],
                                "best_acc":     max(val),
                                "dsi":          1 - processed / (len(trainable) * E),
                                "processed":    processed,
                                "noisy_excl":   float(np.mean([h["noisy_excluded_frac"] for h in history
                                                               if h["excluded_frac"] > 0])) if noisy.any() and
                                                any(h["excluded_frac"] > 0 for h in history) else np.nan,
                                "est_noise":    float(np.nanmean([h["est_negmargin"] for h in history[args.onset:]]))
                                                if len(history) > args.onset else np.nan,
                                "est_gmm":      float(np.nanmean([h["est_gmm"] for h in history[args.onset:]]))
                                                if len(history) > args.onset else np.nan,
                                "budget_mean":  float(np.mean([h["excluded_frac"] for h in history[args.onset:]]))
                                                if len(history) > args.onset else 0.0,
                                "rule_p":       rule["p"],
                                "rule_rho":     rule["rho"],
                                "rule_T":       rule["T"],
                                "p_hat":        float(np.nanmean([h["est_anchored"] for h in history
                                                                  if args.onset - 2 <= h["epoch"] <= args.onset]))
                                                if trusted_loader is not None else np.nan,
                                "wall_minutes": (time() - started) / 60,
                            }
    run_dir.mkdir(parents = True, exist_ok = True)
    with open(run_dir / f"{run_id}.json", "w") as f: dump({**result, "history": history}, f, indent = 2)
    return result

# RESULTS I/O ======================================================================================

def load_results(path: Path) -> pd.DataFrame:
    """# Read results.csv, back-filling columns added after older runs were recorded."""
    df =        pd.read_csv(path, keep_default_na = False, na_values = [""], dtype = {"run_id": str})
    for k, default in EXTRA_DEFAULTS.items():
        df[k] = df[k].fillna(default) if k in df else default
    return df

def append_result(path: Path, result: Dict) -> None:
    """# Append one run, rewriting the file if its columns changed (keeps rows aligned)."""
    row =       pd.DataFrame([result])
    if not path.exists(): row.to_csv(path, index = False); return
    header =    list(pd.read_csv(path, nrows = 0).columns)
    if set(header) == set(row.columns): row[header].to_csv(path, mode = "a", header = False, index = False)
    else:                               pd.concat([load_results(path), row], ignore_index = True).to_csv(path, index = False)

def run_config(args: Namespace) -> Dict:
    """# Hashable run configuration (extras only when non-default, preserving stage-1 run IDs)."""
    return  {k: getattr(args, k) for k in POLICY_KEYS} | {"seed": args.seed} | \
            {k: getattr(args, k) for k, d in EXTRA_DEFAULTS.items() if getattr(args, k) != d}

# SUMMARY ==========================================================================================

def summarize(out_dir: Path) -> str:
    """# Seed-Averaged Results with Deltas vs. the Matched Random Control."""
    df =        load_results(out_dir / "results.csv")
    keys =      [k for k in POLICY_KEYS if k not in ("network", "dataset", "batch_size")] + list(EXTRA_DEFAULTS)
    for c in ("noisy_excl", "est_noise", "est_gmm", "budget_mean", "rule_p", "rule_rho", "p_hat"):
        if c not in df: df[c] = np.nan

    # Compute relative to the longest (reference) schedule: processed / (N x reference epochs).
    ref_epochs =    df["epochs"].max()
    df["compute"] = df["processed"] / ((df["processed"] / (1 - df["dsi"])) / df["epochs"] * ref_epochs)

    # Seed-paired difference vs. the matched random arm (same seed => same init & early data order),
    # which removes most between-seed variance. Same stratification preferred, else any.
    schedule =  ["epochs", "label_noise", "noise_type", "objective", "mode", "prune_fraction", "onset", "ramp", "full_final",
                 "budget", "budget_k", "trusted_size"]
    rand_rows = df[(df["score"] == "random") & (df["max_gap"] == 0)]
    exact_s =   rand_rows.groupby(schedule + ["stratify", "seed"])["final_acc"].mean()
    loose_s =   rand_rows.groupby(schedule + ["seed"])["final_acc"].mean()
    df["paired"] = [
                        row.final_acc - exact_s.get(tuple(row[c] for c in schedule + ["stratify", "seed"]),
                                                    loose_s.get(tuple(row[c] for c in schedule + ["seed"]), np.nan))
                        for _, row in df.iterrows()
                    ]

    agg =       df.groupby(keys, dropna = False).agg(
                    n =             ("seed", "count"),
                    final_acc =     ("final_acc", "mean"),
                    final_std =     ("final_acc", "std"),
                    best_acc =      ("best_acc", "mean"),
                    dsi =           ("dsi", "mean"),
                    minutes =       ("wall_minutes", "mean"),
                    paired =        ("paired", "mean"),
                    paired_sd =     ("paired", "std"),
                    n_pairs =       ("paired", "count"),
                    compute =       ("compute", "mean"),
                    noisy_excl =    ("noisy_excl", "mean"),
                    est_noise =     ("est_noise", "mean"),
                    est_gmm =       ("est_gmm", "mean"),
                    budget_mean =   ("budget_mean", "mean"),
                    rule_p =        ("rule_p", "mean"),
                    rule_rho =      ("rule_rho", "mean"),
                ).reset_index()
    agg["paired_se"] =  agg["paired_sd"] / np.sqrt(agg["n_pairs"])

    # Matched control: plain random arm with the same mode & schedule (same stratification preferred).
    plain =     agg[(agg["score"] == "random") & (agg["max_gap"] == 0)]
    exact =     plain.set_index(schedule + ["stratify"])["final_acc"]
    loose =     plain.groupby(schedule)["final_acc"].mean()
    full =      agg[(agg["score"] == "none") & (agg["epochs"] == ref_epochs) & (agg["objective"] == "ce")] \
                .set_index(["label_noise", "noise_type", "trusted_size"])["final_acc"]
    agg["vs_random"] =  [
                            row.final_acc - exact.get(tuple(row[c] for c in schedule + ["stratify"]),
                                                      loose.get(tuple(row[c] for c in schedule), np.nan))
                            for _, row in agg.iterrows()
                        ]
    agg["vs_full"] =    [row.final_acc - full.get((row.label_noise, row.noise_type, row.trusted_size), np.nan)
                         for _, row in agg.iterrows()]

    cols =      ["label_noise", "noise_type", "objective", "epochs", "score", "refresh", "drop", "stratify", "mode", "temperature", "max_gap",
                 "tilt_start", "budget", "budget_k", "trusted_size", "rule_noise", "prune_fraction", "onset", "ramp", "full_final", "n", "final_acc", "final_std",
                 "best_acc", "vs_random", "paired", "paired_se", "n_pairs", "vs_full", "compute", "noisy_excl", "est_noise",
                 "est_gmm", "budget_mean", "rule_p", "rule_rho", "minutes"]
    table =     agg[cols].sort_values(["label_noise", "noise_type", "compute", "vs_random"], ascending = [True, True, False, False])
    for c in ("final_acc", "final_std", "best_acc", "vs_random", "paired", "paired_se", "vs_full", "compute", "noisy_excl",
              "est_noise", "est_gmm", "budget_mean", "rule_p", "rule_rho"):
        table[c] = (100 * table[c]).round(2)
    table["minutes"] = table["minutes"].round(1)

    text = "\n".join([
        f"{len(df)} runs; accuracies, deltas & DSI in percentage points (deltas are final accuracy).",
        "vs_random: difference from the plain random arm with the same mode & schedule (same stratification if available).",
        "paired / paired_se / n_pairs: same comparison computed seed-by-seed (mean, standard error, number of matched seeds).",
        f"compute: samples processed as % of full data for {ref_epochs} epochs (= 100 - DSI on the reference schedule).",
        "vs_full: difference from full data (cross-entropy) on the reference schedule with the same label noise & noise type.",
        "noisy_excl: % of label-flipped samples excluded per pruned epoch (base rate = prune fraction).",
        "est_noise / est_gmm: mean online noise-rate estimate after onset (negative-margin fraction / GMM high-loss weight); NaN for runs before stage 7.",
        "budget_mean: mean fraction excluded per epoch after onset (adaptive runs vary; fixed runs = prune fraction after ramp).",
        "rule_p / rule_rho: (budget = rule) noise rate fed to the rule (estimated or --rule-noise) and the budget it chose.",
        "", table.to_string(index = False), ""
    ])
    (out_dir / "summary.txt").write_text(text)
    return text

# MAIN =============================================================================================

def main() -> None:
    """# Run One Configuration (Skipping Recorded Runs) or Summarize."""
    args =      parse_arguments()
    out_dir =   Path(args.output) / args.dataset / args.network
    out_dir.mkdir(parents = True, exist_ok = True)
    pd.set_option("display.width", 250)

    if args.summarize: print(summarize(out_dir)); return

    config =    run_config(args)
    run_id =    md5(dumps(config, sort_keys = True).encode()).hexdigest()[:10]
    results =   out_dir / "results.csv"

    if args.stop_after:
        target =    out_dir / "calibration" / f"{run_id}_stop{args.stop_after}.json"
        last_snap = max([int(e) for e in args.snapshot_epochs.split(",") if e.strip() and int(e) <= args.stop_after], default = 0)
        snap_file = out_dir / "calibration" / f"{run_id}_ep{last_snap}.npz"
        if target.exists() and (last_snap == 0 or snap_file.exists()):
            print(f"[{run_id}] Calibration already recorded; skipping."); return
        print(f"[{run_id}] calibration (stop after {args.stop_after}) {config}")
        run(args, run_id, out_dir / "calibration");  print(f"[{run_id}] calibration written to {target}");  return

    if results.exists() and run_id in set(load_results(results)["run_id"]):
        print(f"[{run_id}] Already recorded; skipping.  ({config})"); return

    print(f"[{run_id}] {config}")
    result =    run(args, run_id, out_dir / "runs")
    append_result(results, result)
    print(f"[{run_id}] final acc = {result['final_acc']:.4f}; best = {result['best_acc']:.4f}; DSI = {result['dsi']:.4f}")

if __name__ == "__main__": main()
