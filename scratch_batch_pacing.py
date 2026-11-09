"""# Batch- & Sample-Level Pacing Harness

--unit batch (default): the unit that is pruned / introduced is a mini-batch (described below).
--unit sample: the unit is a single training sample. Active samples are reshuffled into fresh mini-batches
every epoch (standard training); every schedule, criterion, the B-PAS rule and the matching work the same
way per sample. Per-sample criteria (measured on each sample's own training pass):
    accuracy    mean correctness over the sample's last 5 observed epochs
    gradient    exact gradient norm of the final (classification) layer for that sample,
                ||softmax - onehot|| * sqrt(||features||^2 + 1)
    activation  CV over its last --window observed epochs of the sample's mean per-stage activation std
    bpas        |change| of that statistic between its last two observed epochs (B-PAS rule per sample)
    Unseen samples (linear) are probed with a forward pass only (counted as 1/3 of a training pass).

--schedule lifecycle (sample unit only): every sample has its own lifecycle instead of following a global curve.
    introduce   training starts on a random --life-start share; each epoch, at least
                (unseen at start) / --life-intro unseen samples - and as many as were just pruned - are
                introduced, choosing among a probed pool (2 x needed) those the model already gives the highest
                probability to their label ("learnable now").
    prune       a sample correct on --life-k consecutive training passes is retired (learned).
    review      a retired sample is re-checked by a forward pass (1/3 cost) after an interval that starts at
                --life-r0 epochs and doubles every time it is still correct (spaced repetition);
    revive      if a check finds it misclassified (forgotten), it returns to training.  --no-revive disables
                review & revival (ablation). If fewer than --life-floor of the samples are active, the retired
                samples due soonest are revived without a check.
    Savings are whatever the lifecycle produces; fixed shapes are then matched to them (--matched-fmin).

Trains <network> on <dataset> over a FIXED partition of the training set into mini-batches (contents
fixed per seed, order reshuffled every epoch), so every batch has an identity whose learning signals
can be tracked across epochs. The number of active batches follows a pacing schedule; a batch-level
criterion decides WHICH batches are pruned (retired) or introduced / revived.

    unseen  --introduce-->  active  --prune-->  retired  --revive-->  active

## Pacing (--schedule), as a fraction f of all batches
    full        every batch, every epoch (reference).
    linear      incremental introduction: f goes f_min -> f_max linearly; batches are only introduced.
    prune       pruning only: f goes f_max -> f_min linearly; batches are only retired.
    sine        bi-directional: f = mid + amp * sin(2 pi k u)   (starts mid-way, grows first).
    cosine      bi-directional: f = mid + amp * cos(2 pi k u)   (starts at f_max, shrinks first).
    inv-cosine  bi-directional: f = mid - amp * cos(2 pi k u)   (starts at f_min: paces in, then prunes).
    bpas        B-PAS (Alam et al., ICLR 2026), pruning only: at the end of epoch i, every active batch whose
                activation statistic changed by |dX| <= delta(i) since epoch i-1 is pruned PERMANENTLY;
                delta(i) = delta_s * exp(alpha * i), alpha = ln(delta_e / delta_s) / epochs (paper's CIFAR
                default delta_s = 1e-6, delta_e = 5e-5). Savings are whatever the threshold produces.
    dynamic     bi-directional, direction set by the criterion's epoch-level trend: compare this epoch's
                aggregate with the mean of the previous --dyn-window epochs; if it improved by more than
                --dyn-delta (relative), PRUNE, otherwise INTRODUCE. Step = clip(--dyn-gain x |relative
                change|, --dyn-step-min, --dyn-step-max); f stays in [f_min, f_max]; starts at f_max.
    u = (epoch - 1) / (epochs - 1); mid = (f_min + f_max) / 2; amp = (f_max - f_min) / 2; k = --cycles.

## Criterion (--criterion): which batches move (all measured on each batch's own training pass)
    random      uniform.
    bpas        B-PAS's signal: |change| of the activation statistic between the batch's last two observed
                epochs.                                          pruned first: SMALLEST change (converged)
    accuracy    the batch's training accuracy.                  pruned first: HIGHEST (already learned)
    gradient    the batch's gradient norm (all parameters).      pruned first: LOWEST  (little to learn)
    activation  stability of the batch's activations: coefficient of variation, over its last --window
                observed epochs, of the mean per-stage activation std (B-PAS-style).
                                                                 pruned first: LOWEST CV (most stable)
    Introduction / revival takes the opposite end (the batches that most need training). --flip reverses
    both orientations (ablation). --temperature T > 0 samples (Gumbel-top-k) instead of top-k.
    Retired batches keep their last observed values (not re-measured while retired).
    Unseen batches (linear) have no history: with --probe k > 0, a pool of k x the number needed is
    measured by a forward + backward pass without an update (counted as a full training pass);
    activation stability needs several observations, so unseen batches are introduced at random under
    the activation criterion. Probing uses eval mode (batch-norm statistics are not updated).
    Dynamic direction signal: the mean over ALL batches of their latest training accuracy (up = improving),
    gradient norm (down = improving) or activation CV (down = improving). Using every batch, not only the
    active ones, keeps the selection itself from moving the signal (pruning the most accurate batches
    would otherwise read as a stall).

## Matched comparisons
    Every comparison holds the batches, seed, training recipe, criterion and savings fixed and varies only
    the direction pattern. `--matched-fmin` on a recorded run prints the f_min at which the fixed curves
    (all with f_max = 1) spend the same compute: (1 + f_min) / 2 = compute.
    sine, cosine, prune and linear with the default f_min = 0.5, f_max = 1 all train on 75% of the
    batch-epochs (25% savings). `dynamic` sets its own compute; `--matched-fmin` prints the f_min at
    which prune / linear spend the same compute as a recorded dynamic run ((1 + f_min) / 2 = compute).

## Usage (from the gradus repo root)
    python scratch_batch_pacing.py --schedule full --seed 1
    python scratch_batch_pacing.py --schedule cosine --criterion activation --seed 1
    python scratch_batch_pacing.py --schedule dynamic --criterion activation --seed 1 --matched-fmin
    python scratch_batch_pacing.py --summarize

## Outputs (under --output/<dataset>/<network>/)
    runs/<run_id>.json, results.csv, summary.txt
"""

from argparse   import ArgumentParser, Namespace
from hashlib    import md5
from json       import dump, dumps, load
from pathlib    import Path
from time       import time
from typing     import Callable, Dict, List, Optional
import warnings

import numpy    as np
import pandas   as pd

CONFIG_KEYS:    List[str] = [
                                "network", "dataset", "epochs", "batch_size", "seed", "schedule", "f_min", "f_max",
                                "cycles", "criterion", "flip", "temperature", "window", "probe", "dyn_window",
                                "dyn_delta", "dyn_gain", "dyn_step_min", "dyn_step_max"
                            ]
# Keys added in stage 2: they enter the run hash only when non-default (stage-1 run IDs unchanged).
EXTRA_DEFAULTS: Dict =      {"delta_start": 0.0, "delta_end": 0.0, "unit": "batch",
                             "life_start": 0.0, "life_k": 0, "life_r0": 0, "life_intro": 0, "life_floor": 0.0, "life_revive": False}
ACC_WINDOW:     int =       5           # (sample unit) epochs averaged for per-sample accuracy
PROBE_COST:     float =     1 / 3       # (sample unit) forward-only probe, relative to a training pass
UNSEEN, ACTIVE, RETIRED = 0, 1, 2
CRITERIA =      ("accuracy", "gradient", "activation", "bpas")

# ARGUMENTS ========================================================================================

def parse_arguments(argv: Optional[List[str]] = None) -> Namespace:
    """# Parse & Normalize Command Line Arguments."""
    p = ArgumentParser(description = "Batch-level pacing harness.")
    p.add_argument("--network",         type = str,     default = "resnet-18")
    p.add_argument("--dataset",         type = str,     default = "cifar-100")
    p.add_argument("--epochs",          type = int,     default = 100)
    p.add_argument("--seed",            type = int,     default = 1)
    p.add_argument("--batch-size",      type = int,     default = 128)
    p.add_argument("--workers",         type = int,     default = 4)
    p.add_argument("--root",            type = str,     default = ".cache/data")
    p.add_argument("--device",          type = str,     default = "auto", choices = ["auto", "cpu", "cuda"])
    p.add_argument("--output",          type = str,     default = "analysis/batch_pacing")

    p.add_argument("--unit",            type = str,     default = "batch",  choices = ["batch", "sample"])
    p.add_argument("--schedule",        type = str,     default = "full",
                   choices = ["full", "linear", "prune", "sine", "cosine", "inv-cosine", "dynamic", "bpas", "lifecycle"])
    p.add_argument("--f-min",           type = float,   default = 0.5)
    p.add_argument("--f-max",           type = float,   default = 1.0)
    p.add_argument("--cycles",          type = float,   default = 1.0)

    p.add_argument("--criterion",       type = str,     default = "random", choices = ["random", *CRITERIA])
    p.add_argument("--flip",            action = "store_true", help = "Reverse which end is pruned / introduced first.")
    p.add_argument("--temperature",     type = float,   default = 0.0)
    p.add_argument("--window",          type = int,     default = 5,    help = "Observed epochs for activation CV.")
    p.add_argument("--probe",           type = float,   default = 2.0,  help = "Unseen candidate pool = probe x needed (linear).")

    p.add_argument("--delta-start",     type = float,   default = 1e-6, help = "(bpas) delta_s.")
    p.add_argument("--delta-end",       type = float,   default = 5e-5, help = "(bpas) delta_e.")
    p.add_argument("--life-start",      type = float,   default = 0.5,  help = "(lifecycle) share of samples active at epoch 1.")
    p.add_argument("--life-k",          type = int,     default = 3,    help = "(lifecycle) consecutive correct passes = learned.")
    p.add_argument("--life-r0",         type = int,     default = 2,    help = "(lifecycle) first review interval, epochs.")
    p.add_argument("--life-intro",      type = int,     default = 30,   help = "(lifecycle) epochs over which all unseen samples are introduced.")
    p.add_argument("--life-floor",      type = float,   default = 0.2,  help = "(lifecycle) minimum active share.")
    p.add_argument("--no-revive",       action = "store_true", help = "(lifecycle) no review / revival (ablation).")
    p.add_argument("--dyn-window",      type = int,     default = 3)
    p.add_argument("--dyn-delta",       type = float,   default = 0.01)
    p.add_argument("--dyn-gain",        type = float,   default = 1.0)
    p.add_argument("--dyn-step-min",    type = float,   default = 0.01)
    p.add_argument("--dyn-step-max",    type = float,   default = 0.05)

    p.add_argument("--matched-fmin",    action = "store_true", help = "Print the prune/linear f_min matching this (recorded) run's compute.")
    p.add_argument("--save-stats",      action = "store_true",
                   help = "Also save each unit's per-epoch activation statistic to runs/<run_id>_stats.npz (re-runs a "
                          "recorded configuration once if its stats file is missing; the result row is not duplicated).")
    p.add_argument("--summarize",       action = "store_true")
    args = p.parse_args(argv)

    # Normalize fields that cannot matter for the chosen configuration (so configs hash identically).
    if args.schedule == "full":
        args.f_min = args.f_max = 1.0;  args.cycles = 0.0;  args.criterion = "none"
    if args.schedule in ("linear", "prune", "dynamic", "full", "bpas"): args.cycles = 0.0
    if args.schedule == "bpas":
        if args.criterion != "bpas":            p.error("--schedule bpas uses its own criterion: pass --criterion bpas")
        args.f_min = args.f_max = 1.0
    else:                                       args.delta_start = args.delta_end = 0.0
    if args.criterion in ("random", "none"):    args.flip, args.temperature = False, 0.0
    if args.criterion != "activation":          args.window = 0
    if args.schedule != "linear" or args.criterion in ("random", "none", "activation", "bpas"): args.probe = 0.0
    if args.schedule == "dynamic" and args.criterion == "bpas": p.error("dynamic is not defined for the bpas criterion")
    if args.schedule != "dynamic":
        args.dyn_window = 0;    args.dyn_delta = args.dyn_gain = args.dyn_step_min = args.dyn_step_max = 0.0
    if args.schedule == "dynamic" and args.criterion == "random":
        p.error("--schedule dynamic needs a criterion for its direction signal")
    if args.schedule == "lifecycle":
        if args.unit != "sample":               p.error("--schedule lifecycle is defined per sample: pass --unit sample")
        args.criterion, args.f_min, args.f_max, args.probe = "accuracy", 1.0, 1.0, 0.0     # decisions use correctness
        args.life_revive = not args.no_revive
        if args.life_start >= 1:                args.life_intro = 0
    else:
        args.life_start, args.life_k, args.life_r0, args.life_intro, args.life_floor, args.life_revive = 0.0, 0, 0, 0, 0.0, False
    return args

def run_config(args: Namespace) -> Dict:
    return {k: getattr(args, k) for k in CONFIG_KEYS} | {k: getattr(args, k) for k, d in EXTRA_DEFAULTS.items() if getattr(args, k) != d}

def run_hash(config: Dict) -> str:
    return md5(dumps(config, sort_keys = True).encode()).hexdigest()[:10]

# PACING (pure numpy) ==============================================================================

def curve_fraction(u: float, args: Namespace) -> float:
    """# Target fraction of active batches at curve position u in [0, 1] (non-dynamic schedules)."""
    lo, hi =    args.f_min, args.f_max
    if args.schedule == "full":     return 1.0
    if args.schedule == "linear":   return lo + (hi - lo) * u
    if args.schedule == "prune":    return hi + (lo - hi) * u
    w =         np.sin(2 * np.pi * args.cycles * u) if args.schedule == "sine" else np.cos(2 * np.pi * args.cycles * u)
    if args.schedule == "inv-cosine": w = -w
    return float((lo + hi) / 2 + (hi - lo) / 2 * w)

def bpas_threshold(i: int, args: Namespace) -> float:
    """# B-PAS dynamic threshold delta(i) = delta_s * exp(alpha * i), alpha = ln(delta_e / delta_s) / epochs."""
    return float(args.delta_start * np.exp(np.log(args.delta_end / args.delta_start) / args.epochs * i))

def dynamic_step(signal: List[float], higher_is_better: bool, args: Namespace) -> float:
    """# Signed change of f: negative (prune) while the criterion improves, positive (introduce) when it stalls."""
    if len(signal) <= args.dyn_window: return 0.0
    prev =      float(np.mean(signal[-1 - args.dyn_window:-1]))
    rel =       (signal[-1] - prev) / (abs(prev) + 1e-12)
    rel =       rel if higher_is_better else -rel                     # > 0 = improving
    step =      float(np.clip(args.dyn_gain * abs(rel), args.dyn_step_min, args.dyn_step_max))
    return -step if rel > args.dyn_delta else step

# SELECTION (pure numpy) ===========================================================================

def redundancy(criterion: str, acc: np.ndarray, grad: np.ndarray, act_cv: np.ndarray, flip: bool,
               act_delta: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """# Per-batch redundancy: the HIGHEST is pruned first, the LOWEST introduced / revived first."""
    if criterion == "accuracy":     r = acc.astype(np.float64)
    elif criterion == "gradient":   r = -grad.astype(np.float64)
    elif criterion == "activation": r = -act_cv.astype(np.float64)
    elif criterion == "bpas":       r = -act_delta.astype(np.float64)
    else:                           return None
    return -r if flip else r

def pick(candidates: np.ndarray, k: int, key: Optional[np.ndarray], highest: bool, temperature: float,
         rng: np.random.Generator) -> np.ndarray:
    """# Choose k candidates with the highest (or lowest) key; T > 0 samples; NaN keys last, in random order."""
    k =         min(k, len(candidates))
    if k <= 0: return candidates[:0]
    s =         None if key is None else key[candidates].astype(np.float64)
    if s is None or not np.isfinite(s).any(): return rng.choice(candidates, size = k, replace = False)
    z =         (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)
    z =         z if highest else -z
    order =     z / temperature + rng.gumbel(size = len(z)) if temperature > 0 else z + rng.uniform(0, 1e-9, size = len(z))
    nan =       ~np.isfinite(order)
    order[nan] = -1e12 + rng.uniform(0, 1, size = int(nan.sum()))
    return candidates[np.argsort(-order, kind = "stable")[:k]]

def update_membership(status: np.ndarray, target: int, key: Optional[np.ndarray], temperature: float,
                      rng: np.random.Generator, probe: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                      probe_factor: float = 0.0) -> Dict[str, int]:
    """# Move batches between pools so exactly `target` are active. Unseen batches are introduced before
    retired ones are revived. `key` is redundancy (None = random); `probe` scores unseen candidates.
    """
    moved =     {"introduced": 0, "revived": 0, "retired": 0, "probed": 0}
    current =   int((status == ACTIVE).sum())
    if target > current:
        need =      target - current
        unseen =    np.flatnonzero(status == UNSEEN)
        take =      min(need, len(unseen))
        if take:
            pool =  unseen
            k =     key
            if probe is not None and probe_factor > 0 and key is not None:
                pool =  rng.choice(unseen, size = min(len(unseen), int(np.ceil(probe_factor * take))), replace = False)
                k =     np.full(len(status), np.nan);   k[pool] = probe(pool);  moved["probed"] += len(pool)
            elif key is not None:
                k =     None                                        # unseen batches have no history
            chosen =    pick(pool, take, k, False, temperature, rng)
            status[chosen] = ACTIVE;    moved["introduced"] += len(chosen)
        if need > take:
            chosen =    pick(np.flatnonzero(status == RETIRED), need - take, key, False, temperature, rng)
            status[chosen] = ACTIVE;    moved["revived"] += len(chosen)
    elif target < current:
        chosen =    pick(np.flatnonzero(status == ACTIVE), current - target, key, True, temperature, rng)
        status[chosen] = RETIRED;   moved["retired"] += len(chosen)
    return moved

def lifecycle_step(status: np.ndarray, streak: np.ndarray, review_at: np.ndarray, interval: np.ndarray, epoch: int,
                   args: Namespace, rng: np.random.Generator, unseen_start: int,
                   probe: Callable[[np.ndarray], tuple]) -> Dict[str, int]:
    """# One epoch of the per-sample lifecycle (before training): prune learned samples, review retired ones
    and revive the forgotten, introduce learnable unseen samples, enforce the active floor.
    `probe(indices)` returns (correct, true-class probability) from a forward pass."""
    n =         len(status)
    moved =     {"introduced": 0, "revived": 0, "retired": 0, "probed": 0, "reviewed": 0}

    learned =   (status == ACTIVE) & (streak >= args.life_k)
    status[learned] = RETIRED;  interval[learned] = args.life_r0;  review_at[learned] = epoch + args.life_r0
    moved["retired"] = int(learned.sum())

    if args.life_revive:
        due =   np.flatnonzero((status == RETIRED) & (review_at <= epoch) & ~learned)
        if len(due):
            correct, _ = probe(due);    moved["probed"] += len(due);    moved["reviewed"] = len(due)
            forgot =    due[~correct.astype(bool)];   kept = due[correct.astype(bool)]
            status[forgot] = ACTIVE;    streak[forgot] = 0;     moved["revived"] += len(forgot)
            interval[kept] *= 2;        review_at[kept] = epoch + interval[kept]

    unseen =    np.flatnonzero(status == UNSEEN)
    if len(unseen) and args.life_intro > 0:
        need =  min(len(unseen), max(moved["retired"], int(np.ceil(unseen_start / args.life_intro))))
        pool =  rng.choice(unseen, size = min(len(unseen), 2 * need), replace = False)
        _, p_true = probe(pool);    moved["probed"] += len(pool)
        chosen = pool[np.argsort(-p_true, kind = "stable")[:need]]
        status[chosen] = ACTIVE;    streak[chosen] = 0;     moved["introduced"] = len(chosen)

    short =     int(np.ceil(args.life_floor * n)) - int((status == ACTIVE).sum())
    if short > 0:
        retired = np.flatnonzero(status == RETIRED)
        back =  retired[np.argsort(review_at[retired], kind = "stable")[:short]]
        status[back] = ACTIVE;      streak[back] = 0;       moved["revived"] += len(back)
    return moved

def activation_delta(act: np.ndarray, epoch: int) -> np.ndarray:
    """# Per-batch |change| of the activation statistic between its last two OBSERVED epochs (NaN if < 2)."""
    obs =       act[:, :epoch]
    seen =      ~np.isnan(obs)
    rank =      np.cumsum(seen[:, ::-1], axis = 1)[:, ::-1]           # 1 = last observed, 2 = second-to-last
    last =      np.nansum(np.where(seen & (rank == 1), obs, np.nan), axis = 1)
    prev =      np.nansum(np.where(seen & (rank == 2), obs, np.nan), axis = 1)
    out =       np.abs(last - prev)
    out[seen.sum(axis = 1) < 2] = np.nan
    return out

def window_mean(values: np.ndarray, epoch: int, window: int) -> np.ndarray:
    """# Per-unit mean over its last `window` OBSERVED epochs (NaN if never observed)."""
    obs =       values[:, :epoch]
    seen =      ~np.isnan(obs)
    recent =    seen & (np.cumsum(seen[:, ::-1], axis = 1)[:, ::-1] <= window)
    with warnings.catch_warnings(), np.errstate(all = "ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.where(recent, obs, np.nan), axis = 1)

def activation_cv(act: np.ndarray, epoch: int, window: int) -> np.ndarray:
    """# Per-batch CV of the activation statistic over its last `window` OBSERVED epochs (NaN if < 2)."""
    obs =       act[:, :epoch]
    seen =      ~np.isnan(obs)
    recent =    seen & (np.cumsum(seen[:, ::-1], axis = 1)[:, ::-1] <= max(2, window))
    w =         np.where(recent, obs, np.nan)
    with warnings.catch_warnings(), np.errstate(all = "ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        cv =    np.nanstd(w, axis = 1) / np.abs(np.nanmean(w, axis = 1))
    cv[recent.sum(axis = 1) < 2] = np.nan
    return cv

# TRAINING =========================================================================================

def run(args: Namespace, run_id: str, out_dir: Path, rerun: bool = False) -> Dict:
    """# Train One Configuration."""
    from torch                      import Generator, arange, no_grad, randperm, stack, tensor
    from torch.nn.functional        import cross_entropy, softmax
    from torch.optim                import SGD
    from torch.optim.lr_scheduler   import CosineAnnealingLR
    from torch.utils.data           import DataLoader, Dataset, Sampler
    from tqdm                       import tqdm

    from gradus.registration        import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities           import determine_device, set_seed

    class Indexed(Dataset):
        def __init__(self, data):   self.data = data
        def __len__(self):          return len(self.data)
        def __getitem__(self, i):   x, y = self.data[i]; return x, y, i

    class FixedBatches(Sampler):
        """Yields the active batches (fixed contents) in a fresh random order every epoch."""
        def __init__(self, batches, seed):  self.batches, self.active, self.gen = batches, np.arange(len(batches)), Generator().manual_seed(seed)
        def __len__(self):                  return len(self.active)
        def __iter__(self):
            order = randperm(len(self.active), generator = self.gen).numpy()
            return iter([self.batches[b].tolist() for b in self.active[order]])

    set_seed(args.seed);    device = determine_device(args.device)
    rng =                   np.random.default_rng(args.seed)
    E =                     args.epochs

    dataset =               DATASET_REGISTRY.load_dataset(dataset_id = args.dataset, epochs = E, seed = args.seed,
                                                          root = args.root, batch_size = args.batch_size, max_workers = args.workers)
    train =                 dataset.train_data
    N =                     len(train)
    perm =                  np.random.default_rng(40_000 + args.seed).permutation(N)     # fixed partition per seed
    batches =               [perm[i:i + args.batch_size] for i in range(0, N, args.batch_size)]
    B =                     len(batches)
    batch_of =              np.empty(N, dtype = np.int64)
    for b, members in enumerate(batches): batch_of[members] = b

    class ShuffledSamples(Sampler):
        """Yields the active samples reshuffled into fresh mini-batches every epoch."""
        def __init__(self, n, bs, seed):    self.active, self.bs, self.gen = np.arange(n), bs, Generator().manual_seed(seed)
        def __len__(self):                  return (len(self.active) + self.bs - 1) // self.bs
        def __iter__(self):
            order = self.active[randperm(len(self.active), generator = self.gen).numpy()]
            return iter([order[i:i + self.bs].tolist() for i in range(0, len(order), self.bs)])

    sample_unit =           args.unit == "sample"
    U =                     N if sample_unit else B
    sampler =               ShuffledSamples(N, args.batch_size, args.seed) if sample_unit else FixedBatches(batches, args.seed)
    indexed =               Indexed(train)
    loader =                DataLoader(indexed, batch_sampler = sampler, num_workers = args.workers, pin_memory = True,
                                       persistent_workers = args.workers > 0)
    network =               NETWORK_REGISTRY.load_network(network_id = args.network, input_shape = dataset.input_shape,
                                                          num_classes = dataset.num_classes).to(device)
    params =                [p for p in network.parameters() if p.requires_grad]
    optimizer =             SGD(params, lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    scheduler =             CosineAnnealingLR(optimizer, T_max = E)

    acc =                   np.full(U, np.nan)
    grad =                  np.full(U, np.nan)
    act =                   np.full((U, E), np.nan, dtype = np.float32 if sample_unit else np.float64)
    corr =                  np.full((U, E), np.nan, dtype = np.float32) if sample_unit else None
    status =                np.full(U, UNSEEN, dtype = np.int8)

    def measure(x, y):
        """Forward + backward on one batch: (logits, loss, accuracy, gradient norm, activation statistic)."""
        logits, acts =      network(x, return_activations = True)
        loss =              cross_entropy(logits, y)
        loss.backward()
        g =                 stack([p.grad.detach().norm() for p in params if p.grad is not None]).norm().item()
        a =                 float(np.mean([t.float().std().item() for t in acts]))
        return logits, loss, logits.argmax(dim = 1).eq(y).float().mean().item(), g, a

    def measure_samples(x, y):
        """Forward + backward on one batch; per-sample (correct, final-layer gradient norm, activation statistic)."""
        logits, acts =      network(x, return_activations = True)
        loss =              cross_entropy(logits, y)
        loss.backward()
        with no_grad():
            e =             softmax(logits.detach(), dim = 1);  e[arange(len(y), device = y.device), y] -= 1
            h =             acts[-1].float().mean(dim = (2, 3))
            g =             (e.norm(dim = 1) * (h.norm(dim = 1) ** 2 + 1).sqrt()).cpu().numpy()
            a =             stack([t.float().flatten(1).std(dim = 1) for t in acts]).mean(dim = 0).cpu().numpy()
            c =             logits.detach().argmax(dim = 1).eq(y).float().cpu().numpy()
        return c, g, a

    def probe_samples(candidates):
        """Forward-only measurement of unseen samples (eval mode): correctness & final-layer gradient norm."""
        network.eval();     out = np.empty(len(candidates))
        with no_grad():
            for k in range(0, len(candidates), 512):
                part =      candidates[k:k + 512]
                xs, ys =    zip(*[train[i] for i in part])
                x, y =      stack(list(xs)).to(device), tensor(ys).to(device)
                logits, acts = network(x, return_activations = True)
                e =         softmax(logits, dim = 1);  e[arange(len(y), device = y.device), y] -= 1
                h =         acts[-1].float().mean(dim = (2, 3))
                g =         (e.norm(dim = 1) * (h.norm(dim = 1) ** 2 + 1).sqrt()).cpu().numpy()
                c =         logits.argmax(dim = 1).eq(y).float().cpu().numpy()
                out[k:k + len(part)] = redundancy(args.criterion, c, g, np.full(len(part), np.nan), args.flip)
        network.train()
        return out

    def probe_lifecycle(candidates):
        """Forward-only check (eval mode): (correct, probability of the given label) per sample."""
        network.eval();     corr_out = np.empty(len(candidates));   p_out = np.empty(len(candidates))
        with no_grad():
            for k in range(0, len(candidates), 512):
                part =      candidates[k:k + 512]
                xs, ys =    zip(*[train[i] for i in part])
                x, y =      stack(list(xs)).to(device), tensor(ys).to(device)
                prob =      softmax(network(x), dim = 1)
                corr_out[k:k + len(part)] = prob.argmax(dim = 1).eq(y).float().cpu().numpy()
                p_out[k:k + len(part)] =    prob[arange(len(y), device = y.device), y].cpu().numpy()
        network.train()
        return corr_out, p_out

    def probe(candidates):
        """Measure unseen batches without updating the model (eval mode; gradients discarded)."""
        network.eval();     out = np.empty(len(candidates))
        for j, b in enumerate(candidates):
            xs, ys =        zip(*[train[i] for i in batches[b]])
            x =             stack(list(xs)).to(device)
            y =             tensor(ys).to(device)
            optimizer.zero_grad()
            _, _, a_acc, g, _ = measure(x, y)
            optimizer.zero_grad()
            out[j] =        redundancy(args.criterion, np.array([a_acc]), np.array([g]), np.array([np.nan]), args.flip)[0]
        network.train()
        return out

    streak =                np.zeros(U, dtype = np.int64)          # (lifecycle) consecutive correct passes
    review_at =             np.zeros(U, dtype = np.int64)
    interval =              np.zeros(U, dtype = np.int64)
    unseen_start =          0
    history =               []
    processed =             0.0
    f =                     args.f_max
    signal =                []
    started =               time()

    for epoch in range(1, E + 1):

        # Active batches for this epoch.
        moved =             {"introduced": 0, "revived": 0, "retired": 0, "probed": 0}
        if args.schedule == "full":
            status[:] =     ACTIVE
        elif args.schedule == "lifecycle":
            if epoch == 1:
                start =     rng.choice(U, size = int(round(min(1.0, args.life_start) * U)), replace = False)
                status[start] = ACTIVE;     unseen_start = int((status == UNSEEN).sum())
            else:
                moved =     lifecycle_step(status, streak, review_at, interval, epoch, args, rng, unseen_start, probe_lifecycle)
        elif args.schedule == "bpas":
            if epoch == 1: status[:] = ACTIVE
            elif epoch >= 3:        # prune after epoch i = epoch - 1, using the change from epoch i - 1 to i
                i =         epoch - 1
                d =         np.abs(act[:, i - 1] - act[:, i - 2])
                drop =      (status == ACTIVE) & np.isfinite(d) & (d <= bpas_threshold(i, args))
                status[drop] = RETIRED;     moved["retired"] = int(drop.sum())
        else:
            if args.schedule != "dynamic": f = curve_fraction((epoch - 1) / max(1, args.epochs - 1), args)
            if sample_unit: acc = window_mean(corr, epoch - 1, ACC_WINDOW)
            key =           redundancy(args.criterion, acc, grad, activation_cv(act, epoch - 1, args.window), args.flip,
                                       activation_delta(act, epoch - 1) if args.criterion == "bpas" else None)
            moved =         update_membership(status, int(round(f * U)), key, args.temperature, rng,
                                              (probe_samples if sample_unit else probe) if (args.probe > 0 and epoch > 1) else None,
                                              args.probe)
        sampler.active =    np.flatnonzero(status == ACTIVE)
        processed +=        moved["probed"] * (PROBE_COST if sample_unit else args.batch_size)   # batch probe = forward + backward
        moved.setdefault("reviewed", 0)

        network.train();    correct = seen = 0
        for x, y, idx in tqdm(loader, desc = f"[{run_id}] epoch {epoch}/{E}", leave = False):
            x, y =          x.to(device), y.to(device)
            optimizer.zero_grad()
            if sample_unit:
                i =         idx.numpy()
                c, g, a =   measure_samples(x, y)
                optimizer.step()
                corr[i, epoch - 1], grad[i], act[i, epoch - 1] = c, g, a
                streak[i] = np.where(c > 0, streak[i] + 1, 0)
                correct +=  int(c.sum())
            else:
                b =         batch_of[int(idx[0])]
                logits, _, a_acc, g, a =  measure(x, y)
                optimizer.step()
                acc[b], grad[b], act[b, epoch - 1] = a_acc, g, a
                correct +=  int(round(a_acc * y.size(0)))
            seen +=         y.size(0)
            processed +=    y.size(0)
        scheduler.step()

        active =            sampler.active
        if sample_unit: acc = window_mean(corr, epoch, ACC_WINDOW)
        cv =                activation_cv(act, epoch, args.window if args.window else 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            signals =       {"train_acc": correct / max(1, seen), "mean_grad": float(np.nanmean(grad[active])),
                             "mean_act_cv": float(np.nanmean(cv[active])),
                             "all_acc": float(np.nanmean(acc)), "all_grad": float(np.nanmean(grad)), "all_act_cv": float(np.nanmean(cv))}

        # Dynamic: the criterion's epoch-level trend sets the direction of the next change.
        step =              0.0
        if args.schedule == "dynamic":
            name, higher =  {"accuracy": ("all_acc", True), "gradient": ("all_grad", False),
                             "activation": ("all_act_cv", False)}[args.criterion]
            if np.isfinite(signals[name]): signal.append(signals[name])
            step =          dynamic_step(signal, higher, args)
            f =             float(np.clip(f + step, args.f_min, args.f_max))

        network.eval();     hits = total = 0
        with no_grad():
            for x, y in dataset.test_loader:
                hits +=     network(x.to(device)).argmax(dim = 1).eq(y.to(device)).sum().item();  total += y.size(0)

        history.append({"epoch": epoch, "active": int(len(active)), "fraction": len(active) / U, "next_f": f, "step": step,
                        "unseen": int((status == UNSEEN).sum()), "retired": int((status == RETIRED).sum()),
                        **moved, **signals, "test_acc": hits / total})
        print(f"[{run_id}] epoch {epoch:3}/{E}: active = {len(active)}/{U} {args.unit}s (unseen {history[-1]['unseen']}, "
              f"retired {history[-1]['retired']}; +{moved['introduced']} new, +{moved['revived']} revived, -{moved['retired']} pruned)"
              + (f"; next f = {f:.3f} ({'prune' if step < 0 else 'introduce' if step > 0 else 'hold'})" if args.schedule == "dynamic" else "")
              + f"; train acc = {signals['train_acc']:.4f}; grad = {signals['mean_grad']:.3f}; "
              f"act cv = {signals['mean_act_cv']:.4f}; test acc = {hits / total:.4f}")

    test =                  [h["test_acc"] for h in history]
    result =                {
                                "run_id":       run_id,
                                **run_config(args),
                                "trained_epochs": E,
                                "final_acc":    test[-1],
                                "best_acc":     max(test),
                                "compute":      processed / (N * args.epochs),
                                "savings":      1 - processed / (N * args.epochs),
                                "mean_fraction": float(np.mean([h["fraction"] for h in history])),
                                "min_fraction": float(np.min([h["fraction"] for h in history])),
                                "ever_seen":    float(1 - history[-1]["unseen"] / U),
                                "introduced":   int(sum(h["introduced"] for h in history)),
                                "revived":      int(sum(h["revived"] for h in history)),
                                "pruned":       int(sum(h["retired"] for h in history)),
                                "direction_changes": int(np.sum(np.diff(np.sign([h["step"] for h in history if h["step"] != 0])) != 0)),
                                "wall_minutes": (time() - started) / 60,
                            }
    (out_dir / "runs").mkdir(parents = True, exist_ok = True)
    if not rerun:   # a stats-only re-run keeps the original run record
        with open(out_dir / "runs" / f"{run_id}.json", "w") as fh: dump({**result, "history": history}, fh, indent = 2)
    if args.save_stats: np.savez_compressed(out_dir / "runs" / f"{run_id}_stats.npz", act = act.astype(np.float32), status = status)
    return result

# RESULTS I/O ======================================================================================

def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na = False, na_values = [""], dtype = {"run_id": str})

def append_result(path: Path, result: Dict) -> None:
    row =       pd.DataFrame([result])
    if not path.exists(): row.to_csv(path, index = False); return
    header =    list(pd.read_csv(path, nrows = 0).columns)
    if set(header) == set(row.columns): row[header].to_csv(path, mode = "a", header = False, index = False)
    else:                               pd.concat([load_results(path), row], ignore_index = True).to_csv(path, index = False)

# SUMMARY ==========================================================================================

def summarize(out_dir: Path) -> str:
    """# Bi-Directional vs. Pruning-Only & Pacing-Only at Matched Compute, with Full Data as Reference."""
    df =        load_results(out_dir / "results.csv")
    df["unit"] = df["unit"].fillna("batch") if "unit" in df else "batch"
    full =      df[df["schedule"] == "full"]
    full_s =    full.groupby(["unit", "seed"])["final_acc"].mean()
    methods =   df[df["schedule"] != "full"].copy()
    # Lifecycle variants get their own row names: full lifecycle, and the ablations without pacing / revival.
    if "life_start" in methods:
        ls =    methods["life_start"].fillna(0.0);  rv = methods["life_revive"].fillna(False).astype(str).str.lower().eq("true")
        life =  methods["schedule"] == "lifecycle"
        lk =    methods["life_k"].fillna(0).astype(int)
        methods.loc[life, "schedule"] = [f"lifecycle-k{k}" + ("" if s0 < 1 else "-nopace") + ("" if r else "-norevive")
                                         for s0, r, k in zip(ls[life], rv[life], lk[life])]
    methods["vs_full"] = [a - full_s.get((u, sd), np.nan) for u, sd, a in zip(methods["unit"], methods["seed"], methods["final_acc"])]

    # Seed-paired difference vs. the prune-only / pacing-only run with the same criterion, seed and
    # (within 2 points) the same compute.
    for ref in ("prune", "linear"):
        pool =  methods[methods["schedule"].isin(["prune", "bpas"] if ref == "prune" else ["linear"])]
        vals =  []
        for _, r in methods.iterrows():
            if r["schedule"] in ("prune", "linear", "bpas"): vals.append(np.nan); continue
            c =     pool[(pool["criterion"] == r["criterion"]) & (pool["seed"] == r["seed"]) & (pool["unit"] == r["unit"])]
            c =     c[(c["compute"] - r["compute"]).abs() <= 0.02]
            vals.append(r["final_acc"] - c.loc[(c["compute"] - r["compute"]).abs().idxmin(), "final_acc"] if len(c) else np.nan)
        methods[f"vs_{ref}"] = vals

    order =     {"bpas": 0, "prune": 0, "linear": 1, "sine": 2, "cosine": 3, "inv-cosine": 4, "dynamic": 5, "lifecycle": 6}
    agg =       methods.groupby(["unit", "criterion", "schedule", "f_min"], dropna = False).agg(
                    n =             ("seed", "count"),
                    test_acc =      ("final_acc", "mean"),
                    std =           ("final_acc", "std"),
                    vs_full =       ("vs_full", "mean"),
                    savings =       ("savings", "mean"),
                    vs_prune =      ("vs_prune", "mean"),
                    vs_linear =     ("vs_linear", "mean"),
                    dir_changes =   ("direction_changes", "mean"),
                ).reset_index()
    agg["_o"] = agg["schedule"].map(lambda x: order.get(x, 6 if x.startswith("lifecycle") else 9))
    agg =       agg.sort_values(["unit", "criterion", "_o", "f_min"]).drop(columns = "_o")
    for c in ("test_acc", "std", "vs_full", "savings", "vs_prune", "vs_linear"): agg[c] = (100 * agg[c]).round(2)
    agg.loc[agg["schedule"] != "dynamic", "dir_changes"] = np.nan

    blocks = []
    for (unit, crit), g in agg.groupby(["unit", "criterion"], sort = False):
        blocks += [f"-- {unit} level, criterion: {crit} --", g.drop(columns = ["unit", "criterion"]).to_string(index = False), ""]
    text = "\n".join([
        f"{len(df)} runs. Test accuracy and differences in percentage points (mean over seeds; differences are seed-paired).",
        "Reference: full-data training, 100 epochs - " + ("; ".join(f"{u} level ({'fixed batches' if u == 'batch' else 'reshuffled batches'}) = "
        f"{100 * g['final_acc'].mean():.2f} +/- {100 * g['final_acc'].std():.2f} (n={len(g)})" for u, g in full.groupby("unit")) or "not run yet") + ".",
        "savings: % of full-data computation saved (batch passes not trained, net of probes).",
        "vs_full: method - full data. vs_prune / vs_linear: bi-directional - pruning-only (prune, or B-PAS in the bpas block) / pacing-only with the SAME",
        "criterion & seed at the same compute (+-2 points); positive = bi-directional is better.",
        "dir_changes: (dynamic) number of switches between pruning and introducing.",
        "", *blocks
    ])
    (out_dir / "summary.txt").write_text(text)
    return text

# MAIN =============================================================================================

def main() -> None:
    args =      parse_arguments()
    out_dir =   Path(args.output) / args.dataset / args.network
    out_dir.mkdir(parents = True, exist_ok = True)
    pd.set_option("display.width", 250)
    if args.summarize: print(summarize(out_dir)); return

    config =    run_config(args)
    run_id =    run_hash(config)
    if args.matched_fmin:
        rec =   out_dir / "runs" / f"{run_id}.json"
        if not rec.exists(): raise SystemExit(f"[{run_id}] no recorded run for {config}")
        print(f"{max(0.0, 2 * load(open(rec))['compute'] - 1):.3f}"); return

    results =   out_dir / "results.csv"
    recorded =  results.exists() and run_id in set(load_results(results)["run_id"])
    stats =     out_dir / "runs" / f"{run_id}_stats.npz"
    if recorded and not (args.save_stats and not stats.exists()):
        print(f"[{run_id}] Already recorded; skipping.  ({config})"); return
    if recorded: print(f"[{run_id}] Recorded, but re-running once to save per-unit statistics to {stats.name}.")

    print(f"[{run_id}] {config}")
    result =    run(args, run_id, out_dir, rerun = recorded)
    if not recorded: append_result(results, result)
    print(f"[{run_id}] final acc = {result['final_acc']:.4f}; best = {result['best_acc']:.4f}; savings = {result['savings']:.3f}")

if __name__ == "__main__": main()
