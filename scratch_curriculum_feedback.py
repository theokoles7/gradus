"""# Curriculum Feedback and Reentry Diagnostic

One shared model per seed; prospective train/validation split; no test-set use.
After warmup, pair similarly easy examples within each class and randomly
withhold one member of each pair. Stable-view audits measure progress in the
inactive, matched-active, and remaining-active populations. Short, restored-state
branches test reentry against matched and random active-example updates.

This is an exploratory mechanism experiment, not a novel algorithm claim.

Run from the Gradus repository root:
    python scratch_curriculum_feedback.py --device cuda --seeds 1 2

Upload summary.txt, audit_progress.csv, and paired_interventions.csv.
Checkpoints, cohort identities and per-example audits are retained locally.
"""

from argparse       import ArgumentParser, Namespace
from copy           import deepcopy
from datetime       import datetime, timezone
from json           import dump
from math           import cos, pi
from pathlib        import Path
from time           import perf_counter
from typing         import Dict, List

import csv
import platform
import random

import numpy        as np
import torch
from PIL            import Image
from torch.nn.functional import cross_entropy, softmax
from torch.optim    import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as vision
from tqdm           import tqdm

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2761)

# ARGUMENTS ========================================================================================

def parse_arguments() -> Namespace:
    parser = ArgumentParser(description = "Test stale curriculum feedback and paired reentry utility.")
    parser.add_argument("--device",          type = str,   default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--seeds",           type = int,   default = [1, 2], nargs = "+")
    parser.add_argument("--warmup",          type = int,   default = 10)
    parser.add_argument("--withhold-epochs", type = int,   default = 10)
    parser.add_argument("--exclude",         type = float, default = 0.3)
    parser.add_argument("--batch-size",      type = int,   default = 128)
    parser.add_argument("--workers",         type = int,   default = 4)
    parser.add_argument("--audit-per-class", type = int,   default = 10)
    parser.add_argument("--fork-at",         type = int,   default = [1, 5, 10], nargs = "+")
    parser.add_argument("--fork-repeats",    type = int,   default = 3)
    parser.add_argument("--fork-steps",      type = int,   default = 8)
    parser.add_argument("--fork-val-size",   type = int,   default = 2000)
    parser.add_argument("--root",            type = str,   default = ".cache/data")
    parser.add_argument("--output",          type = str,   default = "analysis/curriculum_feedback")
    parser.add_argument("--smoke",           action = "store_true", help = "Tiny synthetic end-to-end check; no download.")
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.warmup, args.withhold_epochs = [1], 1, 1
        args.batch_size, args.workers, args.audit_per_class = 8, 0, 4
        args.fork_at, args.fork_repeats, args.fork_steps, args.fork_val_size = [1], 1, 2, 16
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu --smoke for a local smoke check.")
    if not 0 < args.exclude < 0.5:
        parser.error("--exclude must be strictly between zero and 0.5.")
    if min(args.warmup, args.withhold_epochs, args.batch_size, args.audit_per_class,
           args.fork_repeats, args.fork_steps, args.fork_val_size) < 1 or args.workers < 0:
        parser.error("Counts must be positive; workers may be zero.")
    if args.warmup + args.withhold_epochs >= 100:
        parser.error("This diagnostic retains a 100-epoch LR horizon; use a shorter diagnostic.")
    if any(gap < 1 or gap > args.withhold_epochs for gap in args.fork_at):
        parser.error("--fork-at must lie within the withholding phase.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed labels must be distinct.")
    return args

# DATA: KEYED AUGMENTATION AND A PROSPECTIVE HOLDOUT ==================================================

class Images(Dataset):
    """# Stable Audits or Augmentation Keyed by Seed, Epoch and Original Sample ID."""

    def __init__(self, raw, indices, seed: int, epoch: int = 0, augment: bool = False):
        self.raw, self.indices = raw, np.asarray(indices, dtype = np.int64)
        self.seed, self.epoch, self.augment = seed, epoch, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = int(self.indices[row])
        image = Image.fromarray(self.raw.data[index])
        if self.augment:
            key = (self.seed * 1_000_003 + self.epoch * 10_000_019 + index * 97_409) % (2**63 - 1)
            generator = torch.Generator().manual_seed(key)
            top, left = torch.randint(9, (2,), generator = generator).tolist()
            image = vision.crop(vision.pad(image, 4), top, left, 32, 32)
            if float(torch.rand((), generator = generator)) < 0.5:
                image = vision.hflip(image)
        image = vision.normalize(vision.to_tensor(image), MEAN, STD)
        return image, int(self.raw.targets[index])


def make_loader(raw, indices, seed: int, args: Namespace, epoch: int = 0, augment: bool = False):
    return DataLoader(
        Images(raw, indices, seed, epoch, augment), batch_size = args.batch_size if augment else 256,
        shuffle = augment, num_workers = args.workers, pin_memory = args.device == "cuda",
        generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch), drop_last = False
    )


def split_data(raw, seed: int, smoke: bool):
    labels = np.asarray(raw.targets)
    rng = np.random.default_rng(seed + 9_000)
    train, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = 8 if smoke else 50
        validation.extend(indices[:count]); train.extend(indices[count:])
    return np.asarray(train), np.asarray(validation)


def stratified_subset(indices, labels, count: int, rng):
    classes = np.unique(labels[indices])
    if count < len(classes):
        raise ValueError("Validation panel must contain at least one example per class.")
    chosen = []
    for position, label in enumerate(classes):
        pool = indices[labels[indices] == label]
        take = min(len(pool), count // len(classes) + int(position < count % len(classes)))
        chosen.extend(rng.choice(pool, take, replace = False))
    return np.asarray(chosen, dtype = np.int64)

# MEASUREMENTS ======================================================================================

@torch.no_grad()
def score(model, loader, device, cost: Dict) -> Dict:
    """# All Cached and Fresh Scores Use the Same Unaugmented, Eval-BN Protocol."""
    model.eval()
    records = {key: [] for key in ("loss", "correct", "el2n", "feature_norm")}
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        logits, activations = model(x, return_activations = True)
        probability = softmax(logits, dim = 1)
        probability[torch.arange(len(y), device = device), y] -= 1
        values = {
            "loss": cross_entropy(logits, y, reduction = "none"),
            "correct": logits.argmax(1).eq(y), "el2n": probability.norm(dim = 1),
            "feature_norm": activations[-1].mean(dim = (2, 3)).norm(dim = 1)
        }
        for key, value in values.items():
            records[key].append(value.cpu().numpy())
        cost["audit_images"] += len(y)
    return {key: np.concatenate(values) for key, values in records.items()}


@torch.no_grad()
def validation_loss(model, loader, device, cost: Dict) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        total += float(cross_entropy(model(x), y, reduction = "sum"))
        count += len(y)
    cost["validation_images"] += count
    return total / count


def set_lr(optimizer, epoch: int) -> float:
    # Preserve the original 100-epoch LR horizon; this is not a compressed full-training comparison.
    lr = 0.05 * (1 + cos(pi * (epoch - 1) / 100))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_epoch(model, optimizer, loader, device, cost: Dict, description: str):
    model.train()
    total, correct, count = 0.0, 0, 0
    for x, y in tqdm(loader, desc = description, leave = False):
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(y)
        correct += int(logits.detach().argmax(1).eq(y).sum()); count += len(y)
        cost["training_images"] += len(y); cost["training_updates"] += 1
    return total / count, correct / count

# MATCHED EXPOSURE ASSIGNMENT =======================================================================

def create_cohorts(train, labels, initial: Dict, seed: int, args: Namespace):
    rng = np.random.default_rng(seed + 20_000)
    pairs, remainder, active = [], [], []
    losses = dict(zip(train.tolist(), initial["loss"].tolist()))
    for label in np.unique(labels[train]):
        pool = train[labels[train] == label]
        ordered = np.asarray(sorted(pool, key = lambda index: losses[int(index)]))
        number = int(args.exclude * len(pool))
        if number < args.audit_per_class or len(pool) - 2 * number < args.audit_per_class:
            raise ValueError("Too few examples for the requested exclusion/audit panels.")
        for first, second in ordered[:2 * number].reshape(-1, 2):
            excluded, sentinel = (first, second) if rng.integers(2) else (second, first)
            pairs.append((excluded, sentinel, int(label)))
            active.append(sentinel)
        remainder.extend(ordered[2 * number:]); active.extend(ordered[2 * number:])
    pairs = np.asarray(pairs, dtype = np.int64)
    panels = {"excluded": [], "sentinel": [], "remainder": []}
    for label in np.unique(labels[train]):
        rows = np.flatnonzero(pairs[:, 2] == label)
        chosen = rng.choice(rows, args.audit_per_class, replace = False)
        panels["excluded"].extend(pairs[chosen, 0]); panels["sentinel"].extend(pairs[chosen, 1])
        pool = np.asarray(remainder)[labels[np.asarray(remainder)] == label]
        panels["remainder"].extend(rng.choice(pool, args.audit_per_class, replace = False))
    panels = {name: np.asarray(ids, dtype = np.int64) for name, ids in panels.items()}
    masses = {"excluded": len(pairs) / len(train), "sentinel": len(pairs) / len(train),
              "remainder": len(remainder) / len(train)}
    return np.asarray(active), pairs, panels, masses


def match_current(target_ids, target_records, pool_ids, pool_records, labels):
    """# Greedy Same-Class Matching on Current Loss, EL2N and Feature Norm; Report Match Quality."""
    keys = ("loss", "el2n", "feature_norm")
    targets = np.column_stack([target_records[key] for key in keys])
    pool = np.column_stack([pool_records[key] for key in keys])
    scale = np.maximum(pool.std(axis = 0), 1e-6)
    used, selected, distances = set(), [], []
    for index, values in zip(target_ids, targets):
        available = np.asarray([j for j, item in enumerate(pool_ids)
                                if labels[item] == labels[index] and j not in used], dtype = np.int64)
        if not len(available):
            return None, float("nan")
        distance = np.square((pool[available] - values) / scale).sum(axis = 1)
        winner = int(available[np.argmin(distance)])
        used.add(winner); selected.append(pool_ids[winner]); distances.append(float(np.sqrt(distance.min())))
    return np.asarray(selected), float(np.mean(distances))

# FULL-STATE FORKS ==================================================================================

def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list): return [cpu_copy(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_copy(item) for item in value)
    return deepcopy(value)


def snapshot(model, optimizer, device):
    return {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore(model, optimizer, state):
    model.load_state_dict(state["model"]); optimizer.load_state_dict(deepcopy(state["optimizer"]))
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])


def batch_tensors(raw, ids, seed: int, key: int, device):
    data = Images(raw, ids, seed, key, augment = True)
    values = [data[i] for i in range(len(data))]
    return torch.stack([x for x, _ in values]).to(device), torch.tensor([y for _, y in values], device = device)


def run_forks(model, optimizer, raw, active, panels, records, validation, seed, gap, device, args, cost):
    """# First Actions Differ; Every Later Batch, View and LR Is Shared Within a Replicate."""
    labels = np.asarray(raw.targets)
    state = snapshot(model, optimizer, device)
    base_loss = validation_loss(model, validation, device, cost)
    pool_ids = np.concatenate([panels["sentinel"], panels["remainder"]])
    pool_records = {key: np.concatenate([records["sentinel"][key], records["remainder"][key]])
                    for key in records["excluded"]}
    rows = []
    try:
        for repeat in range(args.fork_repeats):
            rng = np.random.default_rng(seed * 100_003 + gap * 1_009 + repeat)
            count = min(args.batch_size, len(panels["excluded"]))
            selected = rng.choice(len(panels["excluded"]), count, replace = False)
            excluded = panels["excluded"][selected]
            targets = {key: value[selected] for key, value in records["excluded"].items()}
            matched, distance = match_current(excluded, targets, pool_ids, pool_records, labels)
            match_gaps = {}
            if matched is not None:
                positions = {int(index): position for position, index in enumerate(pool_ids)}
                matched_rows = np.asarray([positions[int(index)] for index in matched])
                match_gaps = {f"matched_mean_abs_{key}_gap": float(np.abs(pool_records[key][matched_rows] - targets[key]).mean())
                              for key in ("loss", "el2n", "feature_norm")}
            random_active = []
            for label in labels[excluded]:
                available = active[(labels[active] == label) & ~np.isin(active, random_active)]
                random_active.append(rng.choice(available))
            actions = {"reentry": excluded, "sentinel": panels["sentinel"][selected],
                       "random_active": np.asarray(random_active)}
            if matched is not None: actions["current_matched_active"] = matched

            # Pre-materialize shared suffix views; restore RNG after every branch.
            suffix = [rng.choice(active, count, replace = False) for _ in range(args.fork_steps - 1)]
            prefix_key = 1_000_000 + gap * 10_000 + repeat * 100
            suffix_batches = [batch_tensors(raw, ids, seed, prefix_key + step + 1, device)
                              for step, ids in enumerate(suffix)]
            results = []
            for name, ids in actions.items():
                restore(model, optimizer, state)
                first = batch_tensors(raw, ids, seed, prefix_key, device)
                for step in range(args.fork_steps):
                    x, y = first if step == 0 else suffix_batches[step - 1]
                    model.train(); optimizer.zero_grad(set_to_none = True)
                    cross_entropy(model(x), y).backward(); optimizer.step()
                    cost["branch_images"] += len(y); cost["branch_updates"] += 1
                    if step == 0 or step == args.fork_steps - 1:
                        value = validation_loss(model, validation, device, cost)
                        results.append({"seed": seed, "gap": gap, "repeat": repeat, "action": name,
                                        "horizon": step + 1, "validation_loss": value,
                                        "starting_validation_loss": base_loss,
                                        "matched_feature_distance": distance if name == "current_matched_active" else ""})
                        if name == "current_matched_active": results[-1].update(match_gaps)
            for row in results:
                reference = next(r for r in results if r["action"] == "random_active" and r["horizon"] == row["horizon"])
                row["gain_vs_random_active"] = reference["validation_loss"] - row["validation_loss"]
                rows.append(row)
            print(f"    gap {gap}: completed paired intervention {repeat + 1}/{args.fork_repeats}", flush = True)
    finally:
        restore(model, optimizer, state)
    return rows

# OUTPUTS ==========================================================================================

def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = fields); writer.writeheader(); writer.writerows(rows)


def write_summary(output, progress, interventions, costs = None):
    lines = ["CURRICULUM FEEDBACK / REENTRY: EXPLORATORY MECHANISM CHECK", "",
             "Prospective validation split. Official test set never loaded.",
             "Population estimates use fixed stratified panels and actual cohort weights.",
             "Compare changes within cohorts; their loss levels differ by construction.",
             "Cached-all feedback keeps inactive-panel measurements frozen at exclusion.", "",
             "seed gap active_loss inactive_loss population_loss cached_population inactive_forgetting"]
    for row in progress:
        lines.append(f"{row['seed']:4} {row['gap']:3} {row['active_loss']:11.5f} {row['excluded_loss']:13.5f} "
                     f"{row['population_loss']:15.5f} {row['cached_population_loss']:17.5f} "
                     f"{row['excluded_forgetting_rate']:19.5f}")
    lines += ["", "FEEDBACK CHANGES (negative loss change means improvement)",
              "seed gap population_delta cached_delta opposite_signal paired_excluded_minus_sentinel_delta"]
    for row in progress:
        lines.append(f"{row['seed']:4} {row['gap']:3} {row['population_loss_change']:16.6f} "
                     f"{row['cached_loss_change']:12.6f} {str(row['cached_improves_population_worsens']):>15} "
                     f"{row['paired_excluded_minus_sentinel_loss_change']:36.6f}")
    lines += ["", "INTERVENTIONS: positive gain means LOWER validation CE than the matched random-active action.",
              "seed gap horizon action                   mean_gain repeats"]
    groups = sorted(set((r["seed"], r["gap"], r["horizon"], r["action"]) for r in interventions))
    for seed, gap, horizon, action in groups:
        values = [r["gain_vs_random_active"] for r in interventions
                  if (r["seed"], r["gap"], r["horizon"], r["action"]) == (seed, gap, horizon, action)]
        lines.append(f"{seed:4} {gap:3} {horizon:7} {action:24} {np.mean(values):10.6f} {len(values):7}")
    lines += ["", "Match distances are reported in paired_interventions.csv; poor matching limits residual-history claims.",
              "Forks/checkpoints share states and are correlated; they are not independent training seeds.",
              "Randomized within-pair withholding still has shared-model spillovers.",
              "Inactive forgetting is the fraction correct at exclusion but incorrect now, not a transition count.",
              "Audits do not update weights or BN; branch changes are discarded after every intervention.",
              "Neither forgetting nor audit accuracy alone establishes marginal training utility.",
              "This run does not evaluate a new controller or establish novelty or final test accuracy.",
              "Inspect cost_metadata.json for observation, branch and training costs."]
    if costs:
        lines += ["", "MEASURED COSTS (wall time includes training, diagnostics and checkpoint IO)",
                  "seed main_updates branch_updates training_images audit_images validation_images branch_images wall_minutes"]
        for seed, cost in costs.items():
            lines.append(f"{seed:>4} {cost['training_updates']:12} {cost['branch_updates']:14} "
                         f"{cost['training_images']:15} {cost['audit_images']:12} {cost['validation_images']:17} "
                         f"{cost['branch_images']:13} {cost['wall_minutes']:12.3f}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")

# EXPERIMENT ========================================================================================

def run_seed(raw, seed, args, output, progress, interventions, costs):
    from gradus.registration import NETWORK_REGISTRY

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if args.device == "cuda": torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    labels = np.asarray(raw.targets)
    classes = len(np.unique(labels))
    train, validation_ids = split_data(raw, seed, args.smoke)
    validation_ids = stratified_subset(validation_ids, labels, args.fork_val_size, np.random.default_rng(seed + 5_000))
    validation = make_loader(raw, validation_ids, seed, args)
    model = NETWORK_REGISTRY.load_network(network_id = "resnet-18", input_shape = (3, 32, 32), num_classes = classes).to(device)
    optimizer = SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    cost = {key: 0 for key in ("training_images", "training_updates", "audit_images", "validation_images", "branch_images", "branch_updates")}
    started = perf_counter()
    print(f"[seed {seed}] warmup: {len(train)} train images; held-out development panel: {len(validation_ids)}", flush = True)
    for epoch in range(1, args.warmup + 1):
        lr = set_lr(optimizer, epoch)
        loss, accuracy = train_epoch(model, optimizer, make_loader(raw, train, seed, args, epoch, True),
                                     device, cost, f"seed {seed} warmup {epoch}/{args.warmup}")
        print(f"[seed {seed}] warmup {epoch}: loss={loss:.4f}, train_acc={accuracy:.4f}, lr={lr:.5f}", flush = True)

    initial = score(model, make_loader(raw, train, seed, args), device, cost)
    active, pairs, panels, masses = create_cohorts(train, labels, initial, seed, args)
    np.savez_compressed(output / f"cohorts_seed-{seed}.npz", train_ids = train, validation_ids = validation_ids,
                        pairs = pairs, active_ids = active, **{f"panel_{name}": ids for name, ids in panels.items()})
    loaders = {name: make_loader(raw, ids, seed, args) for name, ids in panels.items()}
    history, baseline, previous = [], None, None
    for gap in range(args.withhold_epochs + 1):
        if gap:
            epoch = args.warmup + gap
            lr = set_lr(optimizer, epoch)
            loss, accuracy = train_epoch(model, optimizer, make_loader(raw, active, seed, args, epoch, True),
                                         device, cost, f"seed {seed} withheld {gap}/{args.withhold_epochs}")
            print(f"[seed {seed}] withholding {gap}: loss={loss:.4f}, train_acc={accuracy:.4f}, lr={lr:.5f}", flush = True)
        records = {name: score(model, loader, device, cost) for name, loader in loaders.items()}
        if baseline is None: baseline = deepcopy(records)
        row = {"seed": seed, "gap": gap, "epoch": args.warmup + gap}
        for name in records:
            row[f"{name}_loss"] = float(records[name]["loss"].mean())
            row[f"{name}_accuracy"] = float(records[name]["correct"].mean())
            initially_correct = baseline[name]["correct"].astype(bool)
            row[f"{name}_forgetting_rate"] = float((~records[name]["correct"][initially_correct]).mean()) \
                                                if initially_correct.any() else float("nan")
        row["population_loss"] = sum(masses[name] * row[f"{name}_loss"] for name in records)
        row["active_loss"] = (masses["sentinel"] * row["sentinel_loss"] + masses["remainder"] * row["remainder_loss"]) / (1 - masses["excluded"])
        row["cached_population_loss"] = (1 - masses["excluded"]) * row["active_loss"] + masses["excluded"] * float(baseline["excluded"]["loss"].mean())
        row["population_loss_change"] = row["population_loss"] - previous["population_loss"] if previous else 0.0
        row["cached_loss_change"] = row["cached_population_loss"] - previous["cached_population_loss"] if previous else 0.0
        row["cached_improves_population_worsens"] = row["cached_loss_change"] < 0 < row["population_loss_change"]
        row["paired_excluded_minus_sentinel_loss_change"] = float(
            ((records["excluded"]["loss"] - baseline["excluded"]["loss"]) -
             (records["sentinel"]["loss"] - baseline["sentinel"]["loss"])).mean())
        progress.append(row); previous = row
        history.append({"gap": gap, **{f"{name}_{key}": value for name, values in records.items() for key, value in values.items()}})
        print(f"[seed {seed}] audit gap {gap}: population={row['population_loss']:.4f}, cached={row['cached_population_loss']:.4f}, "
              f"inactive_forgetting={row['excluded_forgetting_rate']:.3f}", flush = True)
        if gap == 0 or gap in args.fork_at or gap == args.withhold_epochs:
            state = snapshot(model, optimizer, device)
            torch.save({**state, "seed": seed, "gap": gap, "arguments": vars(args)}, output / f"checkpoint_seed-{seed}_gap-{gap}.pt")
            del state
        if gap in args.fork_at:
            interventions.extend(run_forks(model, optimizer, raw, active, panels, records, validation,
                                          seed, gap, device, args, cost))
        write_csv(output / "audit_progress.csv", progress)
        write_csv(output / "paired_interventions.csv", interventions)
        write_summary(output, progress, interventions)
        np.savez_compressed(output / f"audits_seed-{seed}.npz", gaps = np.asarray([r["gap"] for r in history]),
                            **{key: np.stack([r[key] for r in history]) for key in history[0] if key != "gap"})
    if device.type == "cuda": torch.cuda.synchronize()
    cost["wall_minutes"] = (perf_counter() - started) / 60
    cost["cohort_weights"] = masses
    costs[str(seed)] = cost
    with open(output / "cost_metadata.json", "w") as handle: dump(costs, handle, indent = 2)
    write_summary(output, progress, interventions, costs)
    print(f"[seed {seed}] finished in {cost['wall_minutes']:.2f} minutes", flush = True)


def main():
    args = parse_arguments()
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.smoke:
        class Synthetic:
            pass
        raw = Synthetic()
        raw.data = np.random.default_rng(1).integers(0, 256, size = (128, 32, 32, 3), dtype = np.uint8)
        raw.targets = np.repeat(np.arange(4), 32).tolist()
    else:
        # Only the official TRAIN split is opened; the official test set is never evaluated.
        raw = CIFAR100(root = args.root, train = True, download = True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output) / stamp
    output.mkdir(parents = True, exist_ok = False)
    metadata = {"arguments": vars(args), "torch": torch.__version__, "python": platform.python_version(),
                "cuda": torch.version.cuda, "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                "scope": "Randomized exposure and paired action diagnostic; no new controller or novelty claim."}
    with open(output / "metadata.json", "w") as handle: dump(metadata, handle, indent = 2)
    print(f"Output: {output}", flush = True)
    progress, interventions, costs = [], [], {}
    for seed in args.seeds: run_seed(raw, seed, args, output, progress, interventions, costs)
    print(f"\nSend summary.txt, audit_progress.csv and paired_interventions.csv from {output}", flush = True)


if __name__ == "__main__":
    main()
