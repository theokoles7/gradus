"""# Frozen Curriculum Choices: Transfer to New Futures and New Examples

Freeze the selectors from the completed headroom run. Replay their candidate
batches under fresh common suffixes and evaluate on a previously unused panel C.
No new winner is selected using these outcomes. No training prefix is repeated.

Save beside scratch_curriculum_headroom.py and its two supporting scripts.
    python scratch_curriculum_transfer.py --device cuda

This is a transfer diagnostic, not an efficient deployed curriculum or novelty claim.
"""

from argparse       import ArgumentParser
from datetime       import datetime, timezone
from json           import dump, load
from pathlib        import Path
from time           import perf_counter
from types          import SimpleNamespace

import numpy        as np
import pandas       as pd
import torch
from torch.nn.functional import cross_entropy
from torchvision.datasets import CIFAR100

import scratch_curriculum_headroom as headroom

experiment = headroom.experiment

# ARGUMENTS AND COMPLETED RUN ========================================================================

def parse_arguments():
    parser = ArgumentParser(description = "Frozen choices under new suffixes and a fresh development panel.")
    parser.add_argument("--headroom",           type = str, default = "analysis/curriculum_headroom")
    parser.add_argument("--source",             type = str, default = None)
    parser.add_argument("--device",             type = str, default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--seeds",              type = int, default = [1, 2], nargs = "+")
    parser.add_argument("--suffixes",           type = int, default = 4)
    parser.add_argument("--references",         type = int, default = 12)
    parser.add_argument("--panel-size",         type = int, default = 1000)
    parser.add_argument("--output",             type = str, default = "analysis/curriculum_transfer")
    parser.add_argument("--smoke",              action = "store_true")
    args = parser.parse_args()
    if min(args.suffixes, args.references, args.panel_size) < 1: parser.error("Counts must be positive.")
    if args.references < 2: parser.error("Use at least two reference candidates.")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA is unavailable.")
    return args


def locate_headroom(args):
    root = Path(args.headroom)
    candidates = [root] if (root / "metadata.json").exists() else list(root.glob("*"))
    complete = [path for path in candidates if (path / "metadata.json").exists()
                and (path / "candidate_values.csv").exists()
                and all((path / f"identities_seed-{seed}.npz").exists() for seed in args.seeds)]
    if not complete: raise FileNotFoundError("No completed headroom run found. Set --headroom to its exact folder.")
    return max(complete, key = lambda path: (path / "metadata.json").stat().st_mtime)

# FROZEN POLICIES ====================================================================================

def freeze_policies(frame):
    def winner(column, maximum = False):
        index = frame[column].idxmax() if maximum else frame[column].idxmin()
        return int(frame.loc[index, "candidate"])
    return {
        "lookahead_mixture": [winner("loss_A"), winner("loss_B")],
        "alignment_mixture": [winner("alignment_A", True), winner("alignment_B", True)],
        "high_gradient_norm": [winner("gradient_norm", True)],
        "high_loss": [winner("batch_loss", True)], "low_loss": [winner("batch_loss")],
        "high_el2n": [winner("batch_el2n", True)], "low_el2n": [winner("batch_el2n")]
    }


def fresh_panel(raw, train, used, seed, count):
    labels = np.asarray(raw.targets)
    unavailable = np.zeros(len(labels), dtype = bool)
    unavailable[train] = True; unavailable[used] = True
    available = np.flatnonzero(~unavailable)
    classes = np.unique(labels)
    if count % len(classes): raise ValueError("Panel size must be divisible by the number of classes.")
    rng = np.random.default_rng(seed + 84_000)
    chosen = []
    for label in classes:
        pool = available[labels[available] == label]
        take = count // len(classes)
        if len(pool) < take: raise ValueError("Not enough unused holdout images for panel C; reduce --panel-size.")
        chosen.extend(rng.choice(pool, take, replace = False))
    return np.asarray(chosen, dtype = np.int64)

# REPLAY ============================================================================================

def run_seed(raw, source, run, values, seed, gap, horizon, args, output):
    from gradus.registration import NETWORK_REGISTRY

    device = torch.device(args.device)
    state = torch.load(source / f"checkpoint_seed-{seed}_gap-{gap}.pt", map_location = "cpu", weights_only = True)
    original = experiment.Namespace(**state["arguments"]); original.device = args.device
    if bool(original.smoke) != bool(args.smoke): raise ValueError("Source smoke flag and --smoke disagree.")
    with np.load(source / f"cohorts_seed-{seed}.npz", allow_pickle = False) as data:
        train, old_validation = data["train_ids"].copy(), data["validation_ids"].copy()
    with np.load(run / f"identities_seed-{seed}.npz", allow_pickle = False) as data:
        candidates, calibration_ids = data["candidates"].copy(), data["calibration"].copy()
        used = np.concatenate([old_validation, data["panel_A"], data["panel_B"]])
    panel_ids = fresh_panel(raw, train, used, seed, args.panel_size)
    assert set(panel_ids).isdisjoint(train) and set(panel_ids).isdisjoint(used)
    assert np.isin(calibration_ids, train).all() and np.isin(candidates, train).all()
    labels = np.asarray(raw.targets)
    assert all(np.array_equal(labels[row], labels[candidates[0]]) for row in candidates)
    frame = values[(values.seed == seed) & (values.horizon == horizon)].copy()
    if len(frame) != len(candidates) or frame.candidate.nunique() != len(candidates):
        raise ValueError("Candidate identities and source values do not match.")
    if set(frame.candidate.astype(int)) != set(range(len(candidates))):
        raise ValueError("Candidate IDs are not the expected identity-file row indices.")
    policies = freeze_policies(frame)
    reference_rng = np.random.default_rng(seed + 86_000)
    references = reference_rng.choice(len(candidates), min(args.references, len(candidates)), replace = False)
    chosen = sorted(set(references.tolist()).union(*(set(ids) for ids in policies.values())))
    print(f"[seed {seed}] {len(chosen)} unique frozen actions; {len(references)} uniform references; fresh C={len(panel_ids)}", flush = True)
    with open(output / f"frozen_choices_seed-{seed}.json", "w") as handle:
        dump({"policies": policies, "reference_candidates": references.tolist()}, handle, indent = 2)
    np.savez_compressed(output / f"panel_seed-{seed}.npz", panel_C = panel_ids)
    validation = {"C": experiment.make_loader(raw, panel_ids, seed, original)}
    calibration = experiment.DataLoader(
        experiment.Images(raw, calibration_ids, seed), batch_size = original.batch_size,
        shuffle = False, num_workers = original.workers, pin_memory = args.device == "cuda",
        generator = torch.Generator().manual_seed(seed + 64_000)
    )
    model = NETWORK_REGISTRY.load_network(network_id = "resnet-18", input_shape = (3, 32, 32),
                                          num_classes = len(np.unique(labels))).to(device)
    optimizer = experiment.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    cost = {key: 0 for key in ("calibration_images", "validation_images", "branch_images", "branch_updates")}
    first_key = 2_000_000 + gap * 10_000
    first_batches = {index: experiment.batch_tensors(raw, candidates[index], seed, first_key, device) for index in chosen}
    candidate_rows, policy_rows = [], []
    for suffix in range(args.suffixes):
        rng = np.random.default_rng(seed * 1_000_003 + 900_000 + suffix)
        suffix_ids = [rng.choice(train, original.batch_size, replace = False) for _ in range(horizon - 1)]
        suffix_key = 3_000_000 + suffix * 100
        batches = [experiment.batch_tensors(raw, ids, seed, suffix_key + step, device) for step, ids in enumerate(suffix_ids)]
        np.save(output / f"suffix_seed-{seed}_index-{suffix}.npy", np.asarray(suffix_ids, dtype = np.int64))
        outcomes = {}
        for index in chosen:
            experiment.restore(model, optimizer, state)
            for step in range(horizon):
                x, y = first_batches[index] if step == 0 else batches[step - 1]
                model.train(); optimizer.zero_grad(set_to_none = True)
                cross_entropy(model(x), y).backward(); optimizer.step()
                cost["branch_images"] += len(y); cost["branch_updates"] += 1
            value = headroom.endpoints(model, validation, calibration, device, cost)["C"]
            if not np.isfinite(value): raise RuntimeError("Nonfinite endpoint; do not silently discard an action.")
            outcomes[index] = value
            candidate_rows.append({"seed": seed, "suffix": suffix, "candidate": index, "horizon": horizon, "loss_C": value})
        reference_values = np.asarray([outcomes[int(index)] for index in references])
        reference_mean = float(reference_values.mean())
        # Describes finite uniform-reference sampling error, not a policy confidence interval.
        reference_sem = float(np.sqrt((1 - len(references) / len(candidates)) * reference_values.var(ddof = 1) / len(references)))
        for name, indices in policies.items():
            value = float(np.mean([outcomes[index] for index in indices]))
            policy_rows.append({"seed": seed, "suffix": suffix, "policy": name, "horizon": horizon,
                                "loss_C": value, "reference_mean_C": reference_mean,
                                "gain_vs_uniform_references": reference_mean - value,
                                "reference_sampling_sem": reference_sem})
        experiment.write_csv(output / f"candidate_replays_seed-{seed}.csv", candidate_rows)
        experiment.write_csv(output / f"policy_results_seed-{seed}.csv", policy_rows)
        print(f"[seed {seed}] fresh suffix {suffix + 1}/{args.suffixes} complete", flush = True)
    return candidate_rows, policy_rows, cost

# MAIN ==============================================================================================

def main():
    args = parse_arguments()
    run = locate_headroom(args)
    with open(run / "metadata.json") as handle: previous = load(handle)
    args.source = args.source or previous["source"]
    args.gaps = [int(previous["arguments"]["gap"])]
    horizon = int(previous["arguments"]["horizon"])
    if horizon < 2: raise ValueError("Future-transfer testing requires a horizon of at least two updates.")
    source = headroom.replay.locate_source(args)
    with open(source / "metadata.json") as handle: source_metadata = load(handle)
    if args.smoke:
        raw = SimpleNamespace(data = np.random.default_rng(1).integers(0, 256, size = (128, 32, 32, 3), dtype = np.uint8),
                              targets = np.repeat(np.arange(4), 32).tolist())
    else:
        raw = CIFAR100(root = source_metadata["arguments"]["root"], train = True, download = False)
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents = True, exist_ok = False)
    print(f"Headroom input: {run}\nOutput directory: {output}", flush = True)
    values = pd.read_csv(run / "candidate_values.csv")
    all_candidates, all_policies, costs = [], [], {}
    started = perf_counter()
    for seed in args.seeds:
        candidates, policies, cost = run_seed(raw, source, run, values, seed, args.gaps[0], horizon, args, output)
        all_candidates.extend(candidates); all_policies.extend(policies); costs[str(seed)] = cost
        experiment.write_csv(output / "candidate_replays.csv", all_candidates)
        experiment.write_csv(output / "policy_results.csv", all_policies)
    frame = pd.DataFrame(all_policies)
    contrasts = []
    for (seed, suffix), group in frame.groupby(["seed", "suffix"]):
        indexed = group.set_index("policy")
        for comparator in ("alignment_mixture", "high_gradient_norm", "high_loss", "low_loss", "high_el2n", "low_el2n"):
            contrasts.append({"seed": int(seed), "suffix": int(suffix), "comparator": comparator,
                              "lookahead_gain_vs_comparator": float(indexed.loc[comparator, "loss_C"] - indexed.loc["lookahead_mixture", "loss_C"])})
    experiment.write_csv(output / "policy_contrasts.csv", contrasts)
    if args.device == "cuda": torch.cuda.synchronize()
    lines = ["FROZEN CURRICULUM CHOICES: NEW FUTURES / FRESH PANEL C", "",
             f"Input: {run}; smoke: {args.smoke}; wall minutes: {(perf_counter() - started) / 60:.3f}",
             "No reselection on new outcomes. First actions/views fixed; future minibatches/views changed.",
             "Panel C excludes training, the earlier validation panel, and headroom panels A/B.",
             "Lookahead/alignment mixtures average their frozen A and B choices equally; no favorable direction is picked.",
             "Reference is a fixed uniform subset of old candidates, not the exact mean over all64.", "",
             "seed policy                      mean_gain suffix_SD minimum_gain maximum_gain positive_suffixes"]
    for (seed, name), group in frame.groupby(["seed", "policy"]):
        gains = group.gain_vs_uniform_references
        lines.append(f"{seed:4} {name:27} {gains.mean():10.6f} {gains.std(ddof=1):9.6f} "
                     f"{gains.min():12.6f} {gains.max():12.6f} {int((gains > 0).sum())}/{len(gains)}")
    lines += ["", "POSITIVE LOOKAHEAD CONTRAST MEANS LOWER LOSS THAN THE NAMED KNOWN-SIGNAL POLICY"]
    contrast_frame = pd.DataFrame(contrasts)
    for (seed, name), group in contrast_frame.groupby(["seed", "comparator"]):
        lines.append(f"seed {seed}: vs {name:25} mean={group.lookahead_gain_vs_comparator.mean():+.6f}")
    lines += ["", "Suffixes are conditional repeats within the same two trained states, not new training seeds.",
              "Reference-sampling SEM is not a confidence interval for the policy contrast.",
              "A positive transfer result is not final-accuracy evidence or an algorithmic novelty claim.",
              "The original lookahead search was expensive; its cost cannot be omitted from any efficiency claim.",
              f"Output: {output}"]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    with open(output / "metadata.json", "w") as handle:
        dump({"arguments": vars(args), "source": str(source), "headroom": str(run), "costs": costs,
              "torch": torch.__version__}, handle, indent = 2)
    print(summary, flush = True)


if __name__ == "__main__":
    main()
