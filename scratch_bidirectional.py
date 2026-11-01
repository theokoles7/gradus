"""# Bi-Directional Pacing Harness

Trains <network> on <dataset> while the ACTIVE training set changes size over training. Samples
move between three pools:

    unseen  --introduce-->  active  --retire-->  retired  --revive-->  active

The size of the active set follows a schedule curve (as a fraction of the training set); a metric
decides WHICH samples are introduced / retired; a clock decides how fast the curve is traversed.

## Schedules (--schedule)
    full        all data every epoch (reference).
    prune       pruning only:     fraction goes 1 -> f_min linearly; samples are only retired.
    pace        pacing only:      fraction goes f_min -> 1 linearly; samples are only introduced
                                  (incremental introduction, a classic curriculum).
    sine        bi-directional:   f = mid + amp * sin(2 pi k u)  (starts mid-way, grows first).
    cosine      bi-directional:   f = mid + amp * cos(2 pi k u)  (starts at f_max, shrinks first).
                mid = (f_min + f_max) / 2, amp = (f_max - f_min) / 2, k = --cycles, u in [0, 1].
                --invert flips the periodic term (-sin shrinks first; -cos starts at f_min).
    With --clock epoch and a whole number of cycles, every non-full schedule averages
    (f_min + f_max) / 2 of the data, so prune / pace / sine / cosine are compute-matched by design.

## Metric (--metric): which samples move
    random              uniform choice.
    el2n | loss | aum   dynamic: EMA of the per-sample training statistic (aum: high = easy).
                        Unseen samples have no statistics; with --probe k > 0 a candidate pool of
                        k x the number needed is scored by a forward pass of the current model
                        (counted as 1/3 of a training step per sample). Epoch-1 choices are random.
    static:<name>       a precomputed Gradus metric column (e.g. static:spatial-frequency).
    --intro-rank    ascending | descending   introduce / revive the LOWEST (ascending) scores first.
                        el2n / loss ascending = easiest first (classic easy -> hard curriculum).
    --prune-rank    ascending | descending   retire the LOWEST (ascending) scores first.
                        el2n / loss ascending = most-learned first.
    --temperature   0 = deterministic top-k; T > 0 = Gumbel-top-k sampling on standardized score / T.
    --swap s        (sine / cosine) also exchange a fraction s of the active set every epoch: retire
                    s x active by --prune-rank, then introduce as many by --intro-rank.

## Clock (--clock): how fast the curve is traversed
    epoch       u advances uniformly over the epochs after --warmup.
    signal      u advances by speed / (curve epochs - 1) per epoch, where speed = --speed-slow while
                the learning signal is still improving (by >= --min-delta over the best of the last
                --patience epochs) and --speed-fast once it plateaus. u stops at 1 (curve end).
    --signal    train_acc (active set, during the training pass) | train_loss |
                val_acc (a held-out split of --val-size TRAINING samples, never trained on) |
                test_acc (LEAKS THE TEST SET into training decisions: diagnostic only).

## Membership (--membership)
    pools       (default) persistent pools as above: retired samples stay out until revived.
    sampled     every epoch, each sample is included independently with probability pi_i. Samples on the
                prunable side of the score (below the class mean for --prune-rank ascending) share one
                inclusion probability p, the rest are kept (pi = 1), with p set so the expected active
                fraction equals the curve's target; if that needs p < --pi-min, p = --pi-min and the rest
                drop below 1 too. Epoch 1 (no statistics yet) is uniform.
    --rescale   (sampled) weight each active sample's loss by 1 / pi_i, so the expected gradient equals the
                full-data gradient (importance weighting; the mechanism InfoBatch relies on).
    --schedule infobatch   baseline after InfoBatch (Qin et al., ICLR 2024): each epoch, samples whose
                (most recent) loss is below the mean are dropped with probability --ib-ratio and the kept
                ones weighted 1 / (1 - ratio); the last --ib-anneal fraction of epochs use full data.
                Use --metric loss --alpha 1 for the closest match (alpha 1 = most recent loss).

## Budget-aware training (--budget B > 0)
    A fixed SAMPLE budget of B x (training set) x --epochs is spent, and then training stops (mid-epoch if
    needed). The learning rate follows a cosine over samples processed (not epochs), so every arm - and
    `--schedule full --budget B`, the budgeted short schedule - anneals fully within the same compute.
    The curve position u is the fraction of the budget spent (epoch clock), or advances by
    speed x (epoch's share of the budget) (signal clock). InfoBatch anneals over the last --ib-anneal of
    the budget. Compare methods with `vs_budget_full` (same budget & seed, full data).

## Controls (--control), run with the SAME arguments as the method run they control
    replay      replays the method run's exact per-epoch fraction (and swap) trajectory with random
                choices: same compute, same pacing, no metric. Isolates the metric's contribution.
                (sampled membership: every sample included with the realized fraction f, weighted 1 / f
                under --rescale - i.e. unbiased uniform subsampling.)
    short       full data for round(compute x epochs) epochs (own cosine schedule): the compute-matched
                short schedule. Isolates the value of changing the data at all. Runs with the same seed
                and length are shared, so compute-matched methods reuse one short run.

## Usage (from the gradus repo root)
    python scratch_bidirectional.py --schedule cosine --metric el2n --clock signal --signal train_acc
    python scratch_bidirectional.py --schedule cosine --metric el2n --clock signal --signal train_acc --control replay
    python scratch_bidirectional.py --summarize

## Outputs (under --output/<dataset>/<network>/)
    runs/<run_id>.json      config, per-epoch curves (fraction, u, signal, pool sizes, accuracy).
    results.csv             one row per completed run (recorded runs are skipped).
    summary.txt             seed-averaged table with seed-paired deltas vs. full data, replay & short.
"""

from argparse   import ArgumentParser, Namespace
from hashlib    import md5
from json       import dump, dumps, load
from pathlib    import Path
from time       import time
from typing     import Callable, Dict, List, Optional

import numpy    as np
import pandas   as pd

CONFIG_KEYS:    List[str] = [
                                "network", "dataset", "epochs", "batch_size", "seed", "schedule", "f_min", "f_max",
                                "cycles", "invert", "warmup", "swap", "metric", "intro_rank", "prune_rank",
                                "temperature", "stratify", "probe", "clock", "signal", "patience", "min_delta",
                                "speed_slow", "speed_fast", "val_size", "alpha"
                            ]
# Keys added in stage 3: they enter the run hash only when non-default (earlier run IDs are unchanged).
EXTRA_DEFAULTS: Dict = {"membership": "pools", "rescale": False, "pi_min": 0.1, "ib_ratio": 0.0, "ib_anneal": 0.0,
                        "budget": 0.0}
PROBE_COST:     float = 1 / 3       # forward pass ~ 1/3 of a forward + backward training step
UNSEEN, ACTIVE, RETIRED = 0, 1, 2

# ARGUMENTS ========================================================================================

def parse_arguments(argv: Optional[List[str]] = None) -> Namespace:
    """# Parse & Normalize Command Line Arguments."""
    p = ArgumentParser(description = "Bi-directional pacing harness.")
    p.add_argument("--network",         type = str,     default = "resnet-18")
    p.add_argument("--dataset",         type = str,     default = "cifar-100")
    p.add_argument("--epochs",          type = int,     default = 100)
    p.add_argument("--seed",            type = int,     default = 1)
    p.add_argument("--batch-size",      type = int,     default = 128)
    p.add_argument("--workers",         type = int,     default = 4)
    p.add_argument("--root",            type = str,     default = ".cache/data")
    p.add_argument("--device",          type = str,     default = "auto", choices = ["auto", "cpu", "cuda"])
    p.add_argument("--scores",          type = str,     default = None, help = "Static score parquet (default: .cache/scores/<dataset>/metric-scores_seed-1.parquet).")
    p.add_argument("--output",          type = str,     default = "analysis/bidirectional")

    p.add_argument("--schedule",        type = str,     default = "full",   choices = ["full", "prune", "pace", "sine", "cosine", "infobatch"])
    p.add_argument("--f-min",           type = float,   default = 0.5)
    p.add_argument("--f-max",           type = float,   default = 1.0)
    p.add_argument("--cycles",          type = float,   default = 1.0)
    p.add_argument("--invert",          action = "store_true")
    p.add_argument("--warmup",          type = int,     default = 0,    help = "Full-data epochs before the curve starts.")
    p.add_argument("--swap",            type = float,   default = 0.0)
    p.add_argument("--membership",      type = str,     default = "pools",  choices = ["pools", "sampled"])
    p.add_argument("--rescale",         action = "store_true")
    p.add_argument("--pi-min",          type = float,   default = 0.1)
    p.add_argument("--ib-ratio",        type = float,   default = 0.5)
    p.add_argument("--ib-anneal",       type = float,   default = 0.125)
    p.add_argument("--budget",          type = float,   default = 0.0,
                   help = "Sample budget as a fraction of full training; LR cosine over samples; 0 = off.")

    p.add_argument("--metric",          type = str,     default = "random")
    p.add_argument("--intro-rank",      type = str,     default = "ascending",  choices = ["ascending", "descending"])
    p.add_argument("--prune-rank",      type = str,     default = "ascending",  choices = ["ascending", "descending"])
    p.add_argument("--temperature",     type = float,   default = 0.0)
    p.add_argument("--stratify",        type = str,     default = "class",      choices = ["none", "class"])
    p.add_argument("--probe",           type = float,   default = 2.0,  help = "Unseen candidate pool = probe x needed (dynamic metrics; 0 = random).")
    p.add_argument("--alpha",           type = float,   default = 0.5,  help = "EMA weight on the newest observation.")

    p.add_argument("--clock",           type = str,     default = "epoch",      choices = ["epoch", "signal"])
    p.add_argument("--signal",          type = str,     default = "train_acc",  choices = ["train_acc", "train_loss", "val_acc", "test_acc"])
    p.add_argument("--patience",        type = int,     default = 3)
    p.add_argument("--min-delta",       type = float,   default = 0.002)
    p.add_argument("--speed-slow",      type = float,   default = 0.5)
    p.add_argument("--speed-fast",      type = float,   default = 2.0)
    p.add_argument("--val-size",        type = int,     default = 0,    help = "Carve this many training samples into a held-out validation split.")

    p.add_argument("--control",         type = str,     default = "none",   choices = ["none", "replay", "short"])
    p.add_argument("--summarize",       action = "store_true")
    args = p.parse_args(argv)

    # Normalize fields that cannot matter for the chosen configuration (so configs hash identically).
    dynamic = args.metric in ("el2n", "loss", "aum")
    if args.schedule == "full":
        args.f_min = args.f_max = 1.0;  args.cycles = 0.0;  args.invert = False;    args.warmup = args.swap = 0
        args.metric = "none";   args.clock = "epoch"
    if args.schedule in ("prune", "pace"):                  args.cycles, args.invert, args.swap = 0.0, False, 0.0
    if args.schedule == "prune":                            args.intro_rank = "n/a"
    if args.schedule == "pace":                             args.prune_rank = "n/a"
    if args.metric in ("random", "none"):                   args.intro_rank = args.prune_rank = "n/a";  args.temperature = 0.0
    if not dynamic:                                         args.probe = 0.0
    if args.schedule == "prune" and dynamic:                args.probe = 0.0     # nothing is ever introduced
    if args.clock == "epoch":
        args.signal = "n/a";    args.patience = 0;  args.min_delta = args.speed_slow = args.speed_fast = 0.0
    if args.signal == "val_acc" and args.val_size <= 0:     p.error("--signal val_acc needs --val-size > 0")
    if not dynamic:                                         args.alpha = 0.0
    if args.control != "none" and args.schedule == "full":  p.error("--control needs a non-full method run to control")
    if args.budget > 0 and args.control == "short":         p.error("with --budget, the reference is --schedule full --budget B (not --control short)")
    if args.schedule == "infobatch":
        if args.metric in ("random", "none"):               p.error("--schedule infobatch needs a score (use --metric loss)")
        args.membership, args.rescale, args.clock = "sampled", True, "epoch"
        args.f_min = args.f_max = 1.0;  args.cycles = 0.0;  args.invert = False;    args.swap = 0.0;    args.warmup = 0
        args.signal = "n/a";    args.patience = 0;  args.min_delta = args.speed_slow = args.speed_fast = 0.0
        args.intro_rank, args.probe, args.temperature = "n/a", 0.0, 0.0
    else:                                                   args.ib_ratio = args.ib_anneal = 0.0
    if args.membership == "sampled":                        args.swap, args.probe, args.temperature, args.intro_rank = 0.0, 0.0, 0.0, "n/a"
    if args.membership == "pools":                          args.rescale, args.pi_min = False, 0.1
    if args.membership == "sampled" and args.schedule == "pace": p.error("--membership sampled has no unseen pool; use pools for pace")
    return args

def run_config(args: Namespace, control: Optional[str] = None) -> Dict:
    """# Hashable configuration (the control is part of the identity; its source is the same config without it)."""
    return {k: getattr(args, k) for k in CONFIG_KEYS} | {"control": args.control if control is None else control} | \
           {k: getattr(args, k) for k, d in EXTRA_DEFAULTS.items() if getattr(args, k) != d}

def short_config(args: Namespace, epochs: int) -> Dict:
    """# Identity of a short-schedule control: it depends only on the seed and its length."""
    return {k: getattr(args, k) for k in ("network", "dataset", "batch_size", "seed")} | \
           {"control": "short", "short_epochs": epochs}

def run_hash(config: Dict) -> str:
    return md5(dumps(config, sort_keys = True).encode()).hexdigest()[:10]

# SCHEDULE (pure numpy) ============================================================================

def curve_fraction(u: float, args: Namespace) -> float:
    """# Target active fraction at curve position u in [0, 1]."""
    lo, hi =    args.f_min, args.f_max
    if args.schedule == "full":     return 1.0
    if args.schedule == "prune":    return hi + (lo - hi) * u
    if args.schedule == "pace":     return lo + (hi - lo) * u
    w =         np.sin(2 * np.pi * args.cycles * u) if args.schedule == "sine" else np.cos(2 * np.pi * args.cycles * u)
    return float((lo + hi) / 2 + (hi - lo) / 2 * (-w if args.invert else w))

def clock_speed(signal_history: List[float], args: Namespace) -> float:
    """# Curve speed from the learning signal: slow while improving, fast once it plateaus."""
    if args.clock == "epoch": return 1.0
    s =         [-x for x in signal_history] if args.signal == "train_loss" else list(signal_history)
    if len(s) <= args.patience: return args.speed_slow
    improving = s[-1] - max(s[-1 - args.patience:-1]) >= args.min_delta
    return args.speed_slow if improving else args.speed_fast

# MEMBERSHIP (pure numpy) ==========================================================================

def pick(candidates: np.ndarray, k: int, scores: Optional[np.ndarray], rank: str, temperature: float,
         rng: np.random.Generator) -> np.ndarray:
    """# Choose k candidates: lowest scores first (ascending) or highest (descending).

    T = 0 is deterministic; T > 0 samples without replacement ~ exp(+-z / T) (Gumbel-top-k).
    Unscored (NaN) candidates come last, in random order; no scores at all = uniform random.
    """
    k =         min(k, len(candidates))
    if k <= 0: return candidates[:0]
    s =         None if scores is None else scores[candidates].astype(np.float64)
    if s is None or not np.isfinite(s).any(): return rng.choice(candidates, size = k, replace = False)
    z =         (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)
    z =         -z if rank == "ascending" else z                # larger key = chosen first
    key =       z / temperature + rng.gumbel(size = len(z)) if temperature > 0 else z + rng.uniform(0, 1e-9, size = len(z))
    nan =       ~np.isfinite(key)
    key[nan] =  -1e12 + rng.uniform(0, 1, size = int(nan.sum()))
    return candidates[np.argsort(-key, kind = "stable")[:k]]

def update_membership(
    status:         np.ndarray,
    fraction:       float,
    labels:         np.ndarray,
    trainable:      np.ndarray,
    args:           Namespace,
    scores:         Optional[np.ndarray],
    rng:            np.random.Generator,
    probe:          Optional[Callable[[np.ndarray], np.ndarray]] =  None,
    swap:           Optional[float] =                               None,
    random_choice:  bool =                                          False
) -> Dict[str, int]:
    """# Move samples between pools so each group's active count = round(fraction x group size).

    Introduction takes unseen samples first, then revives retired ones. Modifies `status` in place and
    returns move counts (introduced / revived / retired / probed). `random_choice` ignores the metric
    (replay control). `scores` may be updated in place by `probe` for unseen candidates.
    """
    swap =      args.swap if swap is None else swap
    groups =    labels[trainable] if args.stratify == "class" else np.zeros(len(trainable), dtype = np.int64)
    moved =     {"introduced": 0, "revived": 0, "retired": 0, "probed": 0}
    s =         None if random_choice else scores
    T =         args.temperature

    def introduce(members, k, exclude):
        if k <= 0: return
        unseen =    members[(status[members] == UNSEEN) & ~exclude[members]]
        take =      min(k, len(unseen))
        if take:
            pool =  unseen
            if s is not None and probe is not None and args.probe > 0 and not np.isfinite(s[unseen]).all():
                pool =  rng.choice(unseen, size = min(len(unseen), int(np.ceil(args.probe * take))), replace = False)
                s[pool] = probe(pool);  moved["probed"] += len(pool)
            chosen =    pick(pool, take, s, args.intro_rank, T, rng)
            status[chosen] = ACTIVE;    moved["introduced"] += len(chosen)
        rest =      k - take
        if rest > 0:
            retired =   members[(status[members] == RETIRED) & ~exclude[members]]
            chosen =    pick(retired, rest, s, args.intro_rank, T, rng)
            status[chosen] = ACTIVE;    moved["revived"] += len(chosen)

    def retire(members, k, exclude):
        if k <= 0: return
        active =    members[(status[members] == ACTIVE) & ~exclude[members]]
        chosen =    pick(active, k, s, args.prune_rank, T, rng)
        status[chosen] = RETIRED;   moved["retired"] += len(chosen)

    for g in np.unique(groups):
        members =   trainable[groups == g]
        none =      np.zeros(len(status), dtype = bool)
        target =    int(round(fraction * len(members)))
        current =   int((status[members] == ACTIVE).sum())
        if target > current:    introduce(members, target - current, none)
        elif target < current:  retire(members, current - target, none)
        k =         min(int(round(swap * target)), int((status[members] != ACTIVE).sum()))
        if k > 0:   # exchange: newly retired samples cannot return this epoch, and vice versa
            before =    status.copy()
            retire(members, k, none)
            introduce(members, k, (before == ACTIVE) & (status == RETIRED))
    return moved

def inclusion_probs(scores: Optional[np.ndarray], fraction: float, labels: np.ndarray, members: np.ndarray,
                    rank: str, pi_min: float, stratify: str = "class") -> np.ndarray:
    """# Per-sample inclusion probabilities (sampled membership) whose mean is `fraction` within each group.

    The prunable side (scores below the group mean for rank ascending, above it for descending) shares
    probability p; the rest keep pi = 1. If p would fall below pi_min, p = pi_min and the rest share the
    remaining reduction. Unscored samples count as non-prunable. scores None = uniform pi = fraction.
    """
    pi =        np.ones(len(labels))
    if fraction >= 1: return pi
    groups =    labels[members] if stratify == "class" else np.zeros(len(members), dtype = np.int64)
    for g in np.unique(groups):
        m =     members[groups == g]
        cut =   1.0 - fraction                                  # expected fraction to drop
        if scores is None or not np.isfinite(scores[m]).any():
            pi[m] = fraction;   continue
        s =     scores[m]
        mean =  np.nanmean(s)
        low =   np.where(np.isfinite(s), (s < mean) if rank == "ascending" else (s > mean), False)
        f_low = low.mean()
        if f_low > 0 and cut <= f_low * (1 - pi_min):
            pi[m[low]] = 1 - cut / f_low
        else:
            pi[m[low]] = pi_min
            rest =  cut - f_low * (1 - pi_min)
            pi[m[~low]] = max(pi_min, 1 - rest / max(1e-12, 1 - f_low))
    return pi

def infobatch_probs(scores: Optional[np.ndarray], members: np.ndarray, n: int, ratio: float) -> np.ndarray:
    """# InfoBatch: samples with score below the (global) mean are kept with probability 1 - ratio."""
    pi =        np.ones(n)
    if scores is None or not np.isfinite(scores[members]).any(): return pi
    s =         scores[members]
    pi[members[np.isfinite(s) & (s < np.nanmean(s))]] = 1 - ratio
    return pi

# TRAINING =========================================================================================

def carve(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """# Class-stratified boolean mask of `size` held-out samples (fixed per seed)."""
    mask =      np.zeros(len(labels), dtype = bool)
    if size <= 0: return mask
    rng =       np.random.default_rng(30_000 + seed)
    classes =   np.unique(labels)
    per_class = np.full(len(classes), size // len(classes));    per_class[: size % len(classes)] += 1
    for c, k in zip(classes, per_class): mask[rng.choice(np.flatnonzero(labels == c), size = k, replace = False)] = True
    return mask

def run(args: Namespace, run_id: str, out_dir: Path, source: Optional[Dict]) -> Dict:
    """# Train One Configuration."""
    from copy                       import copy
    from torch                      import Generator, no_grad, randperm, tensor as torch_tensor
    from torch.nn.functional        import cross_entropy, one_hot, softmax
    from torch.optim                import SGD
    from torch.optim.lr_scheduler   import CosineAnnealingLR
    from torch.utils.data           import DataLoader, Dataset, Sampler, Subset
    from tqdm                       import tqdm

    from gradus.registration        import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities           import determine_device, set_seed

    class Indexed(Dataset):
        def __init__(self, data):   self.data = data
        def __len__(self):          return len(self.data)
        def __getitem__(self, i):   x, y = self.data[i]; return x, y, i

    class ActiveSampler(Sampler):
        def __init__(self, indices, seed):  self.indices, self.gen = indices, Generator().manual_seed(seed)
        def __len__(self):                  return len(self.indices)
        def __iter__(self):
            order = randperm(len(self.indices), generator = self.gen).numpy()
            return iter(self.indices[order].tolist())

    set_seed(args.seed);    device = determine_device(args.device)
    rng =                   np.random.default_rng(args.seed)
    short =                 args.control == "short"
    E =                     max(1, int(round(source["compute"] * args.epochs))) if short else args.epochs

    dataset =               DATASET_REGISTRY.load_dataset(dataset_id = args.dataset, epochs = E, seed = args.seed,
                                                          root = args.root, batch_size = args.batch_size, max_workers = args.workers)
    train =                 dataset.train_data
    N =                     len(train)
    labels =                np.array(getattr(train, "targets", [train[i][1] for i in range(N)]), dtype = np.int64)
    held =                  carve(labels, args.val_size, args.seed)
    trainable =             np.flatnonzero(~held)

    def eval_loader(view_indices):
        view =              copy(train)
        if hasattr(view, "transform"): view.transform = dataset.test_data.transform
        return DataLoader(Subset(view, view_indices.tolist()), batch_size = 512, shuffle = False, num_workers = args.workers)
    val_loader =            eval_loader(np.flatnonzero(held)) if held.any() else None

    static =                None
    if args.metric.startswith("static:"):
        path =              Path(args.scores or f".cache/scores/{args.dataset}/metric-scores_seed-1.parquet")
        static =            pd.read_parquet(path).set_index("index")[args.metric.split(":", 1)[1]] \
                            .reindex(np.arange(N)).to_numpy(dtype = np.float64)

    sampler =               ActiveSampler(trainable.copy(), args.seed)
    loader =                DataLoader(Indexed(train), batch_size = args.batch_size, sampler = sampler,
                                       num_workers = args.workers, pin_memory = True, persistent_workers = args.workers > 0)
    network =               NETWORK_REGISTRY.load_network(network_id = args.network, input_shape = dataset.input_shape,
                                                          num_classes = dataset.num_classes).to(device)
    optimizer =             SGD(network.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    scheduler =             CosineAnnealingLR(optimizer, T_max = E)

    ema =                   {k: np.full(N, np.nan) for k in ("el2n", "loss", "aum")}
    probed =                np.full(N, np.nan)      # probe scores for samples not yet trained on
    dynamic =               args.metric in ema
    status =                np.full(N, UNSEEN, dtype = np.int8)
    if short or args.schedule == "full": status[trainable] = ACTIVE

    def statistics(z, y):
        onehot =            one_hot(y, num_classes = z.shape[1]).float()
        margin =            z.gather(1, y.unsqueeze(1)).squeeze(1) - z.masked_fill(onehot.bool(), float("-inf")).max(dim = 1).values
        return {"loss": cross_entropy(z, y, reduction = "none").cpu().numpy(),
                "el2n": (softmax(z, dim = 1) - onehot).norm(dim = 1).cpu().numpy(), "aum": margin.cpu().numpy()}

    def probe(indices):
        """Score unseen candidates with a forward pass of the current model (no training)."""
        network.eval();     out = np.empty(len(indices))
        view =              DataLoader(Subset(Indexed(train), indices.tolist()), batch_size = 512, shuffle = False,
                                       num_workers = args.workers)
        pos =               0
        with no_grad():
            for x, y, _ in view:
                v =         statistics(network(x.to(device)), y.to(device))[args.metric]
                out[pos:pos + len(v)] = v;  pos += len(v)
        network.train()
        return out

    def scores():
        if args.metric.startswith("static:"):   return static
        if dynamic:                             return np.where(np.isnan(ema[args.metric]), probed, ema[args.metric])
        return None

    weight =                np.ones(N, dtype = np.float32)     # per-sample loss weight (rescaling)
    sampled =               args.membership == "sampled"
    curve_epochs =          max(2, args.epochs - args.warmup)
    budget =                args.budget * len(trainable) * args.epochs if args.budget > 0 else None
    lr0 =                   optimizer.param_groups[0]["lr"]
    if budget: E =          10 * args.epochs                    # upper bound; the budget ends the run
    u =                     0.0
    signal_history =        []
    history =               []
    processed =             0.0
    started =               time()

    for epoch in range(1, E + 1):

        # Active set for this epoch.
        moved =             {"introduced": 0, "revived": 0, "retired": 0, "probed": 0}
        weight[:] =         1.0
        if not short and args.schedule != "full" and sampled:
            if args.control == "replay":                            # uniform inclusion at the realized fraction
                fraction =  source["history"][min(epoch, len(source["history"])) - 1]["fraction"]
                pi =        np.where(np.isin(np.arange(N), trainable), fraction, 0.0)
            elif args.schedule == "infobatch":
                anneal =    processed >= (1 - args.ib_anneal) * budget if budget else \
                            epoch > args.epochs - int(round(args.ib_anneal * args.epochs))
                pi =        np.ones(N) if (epoch == 1 or anneal) else infobatch_probs(scores(), trainable, N, args.ib_ratio)
                fraction =  float(pi[trainable].mean())
            else:
                fraction =  1.0 if epoch <= args.warmup else curve_fraction(u, args)
                pi =        inclusion_probs(None if epoch == 1 else scores(), fraction, labels, trainable,
                                            args.prune_rank, args.pi_min, args.stratify)
            take =          np.zeros(N, dtype = bool);  take[trainable] = rng.random(len(trainable)) < pi[trainable]
            status[trainable] = np.where(take[trainable], ACTIVE, RETIRED)
            if args.rescale: weight[take] = (1.0 / pi[take]).astype(np.float32)
        elif not short and args.schedule != "full":
            if args.control == "replay":
                h =         source["history"][min(epoch, len(source["history"])) - 1]
                fraction =  h["target"]
                moved =     update_membership(status, fraction, labels, trainable, args, None, rng,
                                              swap = h["swap"], random_choice = True)
            else:
                fraction =  1.0 if epoch <= args.warmup else curve_fraction(u, args)
                s =         None if (dynamic and epoch == 1) else scores()
                moved =     update_membership(status, fraction, labels, trainable, args, s, rng,
                                              probe if (dynamic and epoch > 1) else None,
                                              swap = 0.0 if epoch <= args.warmup or epoch == 1 else None)
                if dynamic and s is not None:   # keep probe scores of samples that are still untrained
                    probed = np.where(np.isnan(ema[args.metric]) & np.isfinite(s), s, probed)
        else: fraction = 1.0
        active =            np.flatnonzero(status == ACTIVE)
        sampler.indices =   active
        processed +=        PROBE_COST * moved["probed"]          # training samples are counted per batch
        epoch_start =       processed

        network.train();    correct = seen = 0;     loss_sum = 0.0
        for x, y, idx in tqdm(loader, desc = f"[{run_id}] epoch {epoch}", leave = False):
            if budget:
                if processed >= budget: break
                for group in optimizer.param_groups: group["lr"] = 0.5 * lr0 * (1 + np.cos(np.pi * processed / budget))
            processed +=    len(y)
            x, y =          x.to(device), y.to(device)
            logits =        network(x)
            per_sample =    cross_entropy(logits, y, reduction = "none")
            w =             torch_tensor(weight[idx.numpy()], device = device)
            optimizer.zero_grad();  (per_sample * w).mean().backward();    optimizer.step()
            with no_grad():
                v =         statistics(logits.detach(), y)
                i =         idx.numpy()
                for k in ema if dynamic else ():
                    prev =  ema[k][i]
                    ema[k][i] = np.where(np.isnan(prev), v[k], args.alpha * v[k] + (1 - args.alpha) * prev)
                correct +=  logits.argmax(dim = 1).eq(y).sum().item();  seen += y.size(0)
                loss_sum += per_sample.sum().item()
        if not budget: scheduler.step()

        network.eval();     hits = total = 0;   val_acc = float("nan")
        with no_grad():
            for x, y in dataset.test_loader:
                hits +=     network(x.to(device)).argmax(dim = 1).eq(y.to(device)).sum().item();  total += y.size(0)
            if val_loader is not None:
                vh = vt =   0
                for x, y in val_loader:
                    vh +=   network(x.to(device)).argmax(dim = 1).eq(y.to(device)).sum().item();  vt += y.size(0)
                val_acc =   vh / vt
        signals =           {"train_acc": correct / max(1, seen), "train_loss": loss_sum / max(1, seen),
                             "val_acc": val_acc, "test_acc": hits / total}

        # Advance the curve clock (after the epoch, from this epoch's signal).
        speed =             1.0
        if not short and args.schedule != "full" and args.control == "none" and epoch > args.warmup:
            if args.clock == "signal": signal_history.append(signals[args.signal])
            speed =         clock_speed(signal_history, args)
            if not budget:              u = min(1.0, u + speed / (curve_epochs - 1))
            elif args.clock == "epoch": u = min(1.0, processed / budget)
            else:                       u = min(1.0, u + speed * (processed - epoch_start) / budget)

        history.append({"epoch": epoch, "target": fraction, "max_weight": float(weight[active].max()) if len(active) else 1.0, "fraction": len(active) / len(trainable), "u": u, "speed": speed,
                        "swap": 0.0 if (epoch <= args.warmup or epoch == 1) else args.swap,
                        "active": int(len(active)), "unseen": int((status[trainable] == UNSEEN).sum()),
                        "retired": int((status[trainable] == RETIRED).sum()), **moved, **signals})
        print(f"[{run_id}] epoch {epoch:3}/{E}: active = {len(active):5}/{len(trainable)} "
              f"(unseen {history[-1]['unseen']}, retired {history[-1]['retired']}; +{moved['introduced']} new, "
              f"+{moved['revived']} revived, -{moved['retired']} retired); u = {u:.3f}; "
              f"train acc = {signals['train_acc']:.4f}; test acc = {signals['test_acc']:.4f}"
              + (f"; budget used = {processed / budget:.3f}" if budget else ""))
        if budget and processed >= budget: break

    test =                  [h["test_acc"] for h in history]
    result =                {
                                "run_id":       run_id,
                                **run_config(args),
                                "source_id":    run_hash(run_config(args, "none")) if args.control != "none" else "",
                                "trained_epochs": len(history),
                                "final_acc":    test[-1],
                                "best_acc":     max(test),
                                "compute":      processed / (len(trainable) * args.epochs),
                                "mean_fraction": float(np.mean([h["fraction"] for h in history])),
                                "ever_seen":    float(1 - history[-1]["unseen"] / len(trainable)),
                                "introduced":   int(sum(h["introduced"] for h in history)),
                                "revived":      int(sum(h["revived"] for h in history)),
                                "retired":      int(sum(h["retired"] for h in history)),
                                "curve_done_epoch": next((h["epoch"] for h in history if h["u"] >= 1.0), np.nan),
                                "wall_minutes": (time() - started) / 60,
                            }
    (out_dir / "runs").mkdir(parents = True, exist_ok = True)
    with open(out_dir / "runs" / f"{run_id}.json", "w") as f: dump({**result, "history": history}, f, indent = 2)
    return result

# RESULTS I/O ======================================================================================

def load_results(path: Path) -> pd.DataFrame:
    """# Read results.csv, back-filling configuration columns added after older runs were recorded."""
    df =        pd.read_csv(path, keep_default_na = False, na_values = [""], dtype = {"run_id": str, "source_id": str})
    for k, default in EXTRA_DEFAULTS.items():
        df[k] = df[k].fillna(default) if k in df else default
    if "rescale" in df: df["rescale"] = df["rescale"].astype(str).str.lower().eq("true")
    return df

def append_result(path: Path, result: Dict) -> None:
    row =       pd.DataFrame([result])
    if not path.exists(): row.to_csv(path, index = False); return
    header =    list(pd.read_csv(path, nrows = 0).columns)
    if set(header) == set(row.columns): row[header].to_csv(path, mode = "a", header = False, index = False)
    else:                               pd.concat([load_results(path), row], ignore_index = True).to_csv(path, index = False)

# SUMMARY ==========================================================================================

def summarize(out_dir: Path) -> str:
    """# Seed-Averaged Results with Seed-Paired Deltas vs. Full Data, Replay & Short Controls."""
    df =        load_results(out_dir / "results.csv")
    df["source_id"] = df["source_id"].fillna("")
    acc =       df.set_index("run_id")["final_acc"]
    full =      df[(df["schedule"] == "full") & (df["epochs"] == df["epochs"].max()) & (df["budget"] == 0)].groupby("seed")["final_acc"].mean()
    bfull =     df[(df["schedule"] == "full") & (df["budget"] > 0)].groupby(["seed", "budget"])["final_acc"].mean()
    methods =   df[(df["control"] == "none") & (df["schedule"] != "full")].copy()
    replay =    df[df["control"] == "replay"].set_index("source_id")["final_acc"]
    methods["vs_replay"] = [a - replay.get(r, np.nan) for r, a in zip(methods["run_id"], methods["final_acc"])]
    short =     df[df["control"] == "short"].groupby(["seed", "trained_epochs"])["final_acc"].mean()
    methods["vs_short"] = [a - short.get((sd, int(round(c * e))), np.nan)
                           for sd, c, e, a in zip(methods["seed"], methods["compute"], methods["epochs"], methods["final_acc"])]
    methods["vs_full"] = [a - full.get(s, np.nan) for s, a in zip(methods["seed"], methods["final_acc"])]
    methods["vs_budget_full"] = [a - bfull.get((s, b), np.nan) if b > 0 else np.nan
                                 for s, b, a in zip(methods["seed"], methods["budget"], methods["final_acc"])]

    # Smoothed short-schedule reference: accuracy vs. trained epochs fitted over every short run and the
    # full-data runs (single short runs are noisy); quadratic once >= 5 distinct lengths are available.
    ref =       pd.concat([df[df["control"] == "short"], df[df["schedule"] == "full"]])
    lengths =   ref["trained_epochs"].nunique()
    fit =       np.polyfit(ref["trained_epochs"], ref["final_acc"], 2 if lengths >= 5 else 1) if lengths >= 2 else None
    methods["vs_short_fit"] = [a - np.polyval(fit, c * e) if fit is not None else np.nan
                               for c, e, a in zip(methods["compute"], methods["epochs"], methods["final_acc"])]

    keys =      [k for k in CONFIG_KEYS if k not in ("network", "dataset", "batch_size", "seed", "epochs")] + list(EXTRA_DEFAULTS)
    agg =       methods.groupby(keys, dropna = False).agg(
                    n =             ("seed", "count"),
                    final_acc =     ("final_acc", "mean"),
                    final_std =     ("final_acc", "std"),
                    vs_full =       ("vs_full", "mean"),
                    vs_replay =     ("vs_replay", "mean"),
                    replay_se =     ("vs_replay", lambda x: x.std() / np.sqrt(x.count()) if x.count() > 1 else np.nan),
                    n_replay =      ("vs_replay", "count"),
                    vs_short =      ("vs_short", "mean"),
                    n_short =       ("vs_short", "count"),
                    vs_short_fit =  ("vs_short_fit", "mean"),
                    vs_budget_full = ("vs_budget_full", "mean"),
                    budget_se =     ("vs_budget_full", lambda x: x.std() / np.sqrt(x.count()) if x.count() > 1 else np.nan),
                    compute =       ("compute", "mean"),
                    ever_seen =     ("ever_seen", "mean"),
                    revived =       ("revived", "mean"),
                    curve_done =    ("curve_done_epoch", "mean"),
                    minutes =       ("wall_minutes", "mean"),
                ).reset_index()
    # Hide fields that only take their default or their normalized-away value.
    quiet =     {"cycles": {0.0, 1.0}, "probe": {0.0, 2.0}, "patience": {0, 3}, "min_delta": {0.0, 0.002},
                 "speed_slow": {0.0, 0.5}, "speed_fast": {0.0, 2.0}, "alpha": {0.0, 0.5}, "f_min": {0.5}, "f_max": {1.0},
                 "warmup": {0}, "swap": {0.0}, "temperature": {0.0}, "val_size": {0}, "stratify": {"class"}, "invert": {False},
                 "pi_min": {0.1}}
    varying =   [k for k in keys if agg[k].nunique(dropna = False) > 1 and not set(agg[k].unique()) <= quiet.get(k, set())]
    cols =      varying + ["n", "final_acc", "final_std", "vs_full", "vs_replay", "replay_se", "n_replay", "vs_short",
                           "n_short", "vs_short_fit", "vs_budget_full", "budget_se", "compute", "ever_seen", "revived", "curve_done", "minutes"]
    table =     agg[cols].sort_values(varying[:3] if varying else ["final_acc"])
    for c in ("final_acc", "final_std", "vs_full", "vs_replay", "replay_se", "vs_short", "vs_short_fit", "vs_budget_full", "budget_se",
              "compute", "ever_seen"):
        table[c] = (100 * table[c]).round(2)
    table["minutes"] = table["minutes"].round(1)

    refs =      df[df["schedule"] == "full"].groupby(["budget", "trained_epochs"])["final_acc"].agg(["mean", "std", "count"]).mul([100, 100, 1]).round(2)
    shorts =    df[df["control"] == "short"].groupby("trained_epochs")["final_acc"].agg(["mean", "count"]).mul([100, 1]).round(2)
    text = "\n".join([
        f"{len(df)} runs ({len(methods)} method runs). Accuracies and deltas in percentage points (final test accuracy).",
        "vs_full: method - full data (100 epochs), seed-paired by seed.",
        "vs_replay: method - its replay control (same per-epoch data fraction, random choices); replay_se = standard error over seeds.",
        "vs_short: method - its compute-matched short schedule (full data for compute x epochs).",
        "vs_short_fit: method - a curve fitted through ALL short & full runs, at the method's compute (less noisy than vs_short).",
        "vs_budget_full / budget_se: (--budget runs) method - full data under the same budget & seed (the budgeted short schedule), seed-paired.",
        "compute: samples processed (incl. probe passes at 1/3) as % of full data for the reference epochs.",
        "ever_seen: % of the training set introduced at least once. revived: samples brought back from the retired pool.",
        "curve_done: epoch at which the curve clock reached its end (signal clock; NaN = never).",
        f"Only configuration fields that vary are shown: {', '.join(varying)}.",
        "", "== full-data reference ==", refs.to_string(), "",
        "== compute-matched short schedules ==", shorts.to_string() if len(shorts) else "(none yet)", "",
        "== methods ==", table.to_string(index = False), ""
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
    results =   out_dir / "results.csv"
    if results.exists() and run_id in set(load_results(results)["run_id"]):
        print(f"[{run_id}] Already recorded; skipping.  ({config})"); return
    source =    None
    if args.control != "none":
        src =   out_dir / "runs" / f"{run_hash(run_config(args, 'none'))}.json"
        if not src.exists(): raise SystemExit(f"[{run_id}] {args.control} control: source run {src.name} not found - run the method first.")
        source = load(open(src))
        if args.control == "short":
            config =    short_config(args, max(1, int(round(source["compute"] * args.epochs))))
            run_id =    run_hash(config)
            if results.exists() and run_id in set(load_results(results)["run_id"]):
                print(f"[{run_id}] Short schedule of {config['short_epochs']} epochs already recorded; skipping."); return

    print(f"[{run_id}] {config}")
    result =    run(args, run_id, out_dir, source)
    append_result(results, result)
    print(f"[{run_id}] final acc = {result['final_acc']:.4f}; best = {result['best_acc']:.4f}; compute = {result['compute']:.3f}")

if __name__ == "__main__": main()
