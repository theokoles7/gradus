"""# Reentry Endpoint Audit: Native vs Recalibrated BatchNorm

Replays saved paired interventions without repeating warmup or withholding.
Each endpoint is measured normally, then with branch-specific BatchNorm
recalibration on the SAME fixed training-only images. Calibration is discarded
before any next update. No official test data are opened.

Save beside scratch_curriculum_feedback.py in the Gradus repository root.
    python scratch_curriculum_bn_audit.py --device cuda

This is a measurement sensitivity check, not a new curriculum or a training win.
"""

from argparse       import ArgumentParser
from datetime       import datetime, timezone
from json           import dump, load
from pathlib        import Path
from time           import perf_counter
from types          import SimpleNamespace

import csv

import numpy        as np
import torch
from torch.nn.functional import cross_entropy
from torchvision.datasets import CIFAR100

import scratch_curriculum_feedback as experiment

# ARGUMENTS AND SOURCE RUN ===========================================================================

def parse_arguments():
    parser = ArgumentParser(description = "Audit reentry validation endpoints using saved checkpoints.")
    parser.add_argument("--source",             type = str, default = "analysis/curriculum_feedback")
    parser.add_argument("--device",             type = str, default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--seeds",              type = int, default = [1, 2], nargs = "+")
    parser.add_argument("--gaps",               type = int, default = [10], nargs = "+")
    parser.add_argument("--calibration-images", type = int, default = 2048)
    parser.add_argument("--output",             type = str, default = "analysis/curriculum_bn_audit")
    parser.add_argument("--smoke",              action = "store_true", help = "Replay an existing synthetic smoke checkpoint.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA is unavailable.")
    if min(args.calibration_images, *args.gaps) < 1: parser.error("Counts/gaps must be positive.")
    return args


def locate_source(args):
    requested = Path(args.source)
    candidates = [requested] if (requested / "metadata.json").exists() else list(requested.glob("*"))
    complete = [path for path in candidates if (path / "metadata.json").exists()
                and all((path / f"checkpoint_seed-{seed}_gap-{gap}.pt").exists()
                        and (path / f"cohorts_seed-{seed}.npz").exists()
                        for seed in args.seeds for gap in args.gaps)]
    if not complete:
        raise FileNotFoundError("No matching checkpoint run found; set --source to the previous run directory.")
    selected = max(complete, key = lambda path: (path / "metadata.json").stat().st_mtime)
    print(f"Source run: {selected}", flush = True)
    return selected

# ENDPOINT MEASUREMENT ===============================================================================

def measure(model, validation, calibration, device, cost):
    """# Recalibration Changes Only Measurement Buffers; Restore Them Before Returning."""
    native = experiment.validation_loss(model, validation, device, cost)
    normalization = [layer for layer in model.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [(layer, {key: value.detach().clone() for key, value in layer.named_buffers(recurse = False)}, layer.momentum)
             for layer in normalization]
    try:
        model.eval()
        for layer in normalization:
            layer.reset_running_stats(); layer.momentum = None; layer.train()
        with torch.no_grad():
            for x, _ in calibration:
                model(x.to(device, non_blocking = True))
                cost["calibration_images"] += len(x)
        calibrated = experiment.validation_loss(model, validation, device, cost)
    finally:
        for layer, buffers, momentum in saved:
            for name, value in buffers.items(): getattr(layer, name).copy_(value)
            layer.momentum = momentum
        model.eval()
    return {"native": native, "recalibrated": calibrated}

# REPLAY ============================================================================================

def replay(raw, source, seed, gap, args, output):
    from gradus.registration import NETWORK_REGISTRY

    device = torch.device(args.device)
    state = torch.load(source / f"checkpoint_seed-{seed}_gap-{gap}.pt", map_location = "cpu", weights_only = True)
    original = experiment.Namespace(**state["arguments"])
    original.device = args.device
    if bool(original.smoke) != bool(args.smoke): raise ValueError("Source smoke flag and --smoke must agree.")
    labels = np.asarray(raw.targets)
    with np.load(source / f"cohorts_seed-{seed}.npz", allow_pickle = False) as identities:
        train, active, validation_ids = [identities[name].copy() for name in ("train_ids", "active_ids", "validation_ids")]
        panels = {name: identities[f"panel_{name}"].copy() for name in ("excluded", "sentinel", "remainder")}
    if not set(train).isdisjoint(validation_ids): raise RuntimeError("Training/validation IDs overlap.")
    model = NETWORK_REGISTRY.load_network(network_id = "resnet-18", input_shape = (3, 32, 32), num_classes = len(np.unique(labels))).to(device)
    optimizer = experiment.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    experiment.restore(model, optimizer, state)
    cost = {"audit_images": 0, "validation_images": 0, "calibration_images": 0, "branch_images": 0, "branch_updates": 0}
    validation = experiment.make_loader(raw, validation_ids, seed, original)
    calibration_rng = np.random.default_rng(seed + 61_000)
    calibration_ids = experiment.stratified_subset(train, labels, args.calibration_images, calibration_rng)
    # Mix classes once; every branch receives the same fixed, full-size batches.
    calibration_ids = calibration_rng.permutation(calibration_ids)
    calibration_ids = calibration_ids[:len(calibration_ids) // original.batch_size * original.batch_size]
    if not len(calibration_ids): raise ValueError("Calibration needs at least one full training-size batch.")
    calibration = experiment.DataLoader(
        experiment.Images(raw, calibration_ids, seed), batch_size = original.batch_size,
        shuffle = False, num_workers = original.workers, pin_memory = args.device == "cuda",
        generator = torch.Generator().manual_seed(seed + 64_000)
    )
    baseline = measure(model, validation, calibration, device, cost)
    records = {name: experiment.score(model, experiment.make_loader(raw, ids, seed, original), device, cost)
               for name, ids in panels.items()}
    pool_ids = np.concatenate([panels["sentinel"], panels["remainder"]])
    pool_records = {key: np.concatenate([records["sentinel"][key], records["remainder"][key]])
                    for key in records["excluded"]}
    rows = []
    for repeat in range(original.fork_repeats):
        # Exactly reproduce the original candidate selection and suffix IDs.
        rng = np.random.default_rng(seed * 100_003 + gap * 1_009 + repeat)
        count = min(original.batch_size, len(panels["excluded"]))
        selected = rng.choice(len(panels["excluded"]), count, replace = False)
        excluded = panels["excluded"][selected]
        target = {key: value[selected] for key, value in records["excluded"].items()}
        matched, distance = experiment.match_current(excluded, target, pool_ids, pool_records, labels)
        random_active = []
        for label in labels[excluded]:
            available = active[(labels[active] == label) & ~np.isin(active, random_active)]
            random_active.append(rng.choice(available))
        actions = {"reentry": excluded, "sentinel": panels["sentinel"][selected],
                   "random_active": np.asarray(random_active)}
        if matched is not None: actions["current_matched_active"] = matched
        suffix = [rng.choice(active, count, replace = False) for _ in range(original.fork_steps - 1)]
        key = 1_000_000 + gap * 10_000 + repeat * 100
        batches = [experiment.batch_tensors(raw, ids, seed, key + step + 1, device)
                   for step, ids in enumerate(suffix)]
        paired = []
        for name, ids in actions.items():
            experiment.restore(model, optimizer, state)
            first = experiment.batch_tensors(raw, ids, seed, key, device)
            for step in range(original.fork_steps):
                x, y = first if step == 0 else batches[step - 1]
                model.train(); optimizer.zero_grad(set_to_none = True)
                cross_entropy(model(x), y).backward(); optimizer.step()
                cost["branch_images"] += len(y); cost["branch_updates"] += 1
                if step == 0 or step == original.fork_steps - 1:
                    endpoints = measure(model, validation, calibration, device, cost)
                    for regime, value in endpoints.items():
                        paired.append({"seed": seed, "gap": gap, "repeat": repeat, "action": name,
                                       "horizon": step + 1, "bn_protocol": regime, "validation_loss": value,
                                       "starting_validation_loss": baseline[regime],
                                       "matched_feature_distance": distance if name == "current_matched_active" else ""})
        for row in paired:
            reference = next(item for item in paired if item["action"] == "random_active"
                             and item["horizon"] == row["horizon"] and item["bn_protocol"] == row["bn_protocol"])
            row["gain_vs_random_active"] = reference["validation_loss"] - row["validation_loss"]
        rows.extend(paired)
        experiment.write_csv(output / "endpoint_replay.csv", rows)
        print(f"[seed {seed}, gap {gap}] replay {repeat + 1}/{original.fork_repeats} complete", flush = True)
    return rows, cost

# MAIN ==============================================================================================

def main():
    args = parse_arguments()
    source = locate_source(args)
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output) / stamp; output.mkdir(parents = True, exist_ok = False)
    rows, costs = [], {}
    started = perf_counter()
    for seed in args.seeds:
        for gap in args.gaps:
            result, cost = replay(raw, source, seed, gap, args, output)
            rows.extend(result); costs[f"seed-{seed}_gap-{gap}"] = cost
            experiment.write_csv(output / "endpoint_replay.csv", rows)
    if args.device == "cuda": torch.cuda.synchronize()
    replay_error = None
    original_results = source / "paired_interventions.csv"
    if original_results.exists():
        with open(original_results, newline = "") as handle:
            reference = {(int(r["seed"]), int(r["gap"]), int(r["repeat"]), r["action"], int(r["horizon"])):
                         float(r["validation_loss"]) for r in csv.DictReader(handle)}
        errors = [abs(r["validation_loss"] - reference[(r["seed"], r["gap"], r["repeat"], r["action"], r["horizon"])])
                  for r in rows if r["bn_protocol"] == "native"
                  and (r["seed"], r["gap"], r["repeat"], r["action"], r["horizon"]) in reference]
        if errors: replay_error = max(errors)
    lines = ["REENTRY ENDPOINT SENSITIVITY: NATIVE / RECALIBRATED BATCHNORM", "",
             f"Source: {source}", f"Synthetic smoke run: {args.smoke}", f"Wall minutes: {(perf_counter() - started) / 60:.3f}",
             "No warmup/withholding repeated; no test data used.",
             "Recalibration uses the same fixed training-only images for all branches.",
             f"Maximum native endpoint difference from original run: {replay_error}",
             "Positive gain means lower validation CE than random-active at the same endpoint.", "",
             "seed gap horizon BN_protocol   action                    mean_gain"]
    groups = sorted(set((r["seed"], r["gap"], r["horizon"], r["bn_protocol"], r["action"]) for r in rows))
    for seed, gap, horizon, regime, action in groups:
        values = [r["gain_vs_random_active"] for r in rows
                  if (r["seed"], r["gap"], r["horizon"], r["bn_protocol"], r["action"]) == (seed, gap, horizon, regime, action)]
        lines.append(f"{seed:4} {gap:3} {horizon:7} {regime:13} {action:25} {np.mean(values):10.6f}")
    lines += ["", "Native endpoints retain the original complete-state objective; recalibrated endpoints change the measurement protocol.",
              "A change of sign indicates sensitivity, not proof the original result was invalid or a new curriculum is useful.",
              "If effects remain negative, close this reentry hypothesis instead of expanding a lifecycle sweep.",
              "Repeated branches are correlated; uncertainty must not treat them as independent training seeds.",
              f"Output: {output}"]
    if replay_error is not None and replay_error > 1e-4:
        lines += ["WARNING: native replay differs from the original; investigate reproducibility before interpreting a protocol difference."]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    with open(output / "metadata.json", "w") as handle:
        dump({"arguments": vars(args), "source": str(source), "costs": costs, "torch": torch.__version__}, handle, indent = 2)
    print(summary, flush = True)


if __name__ == "__main__":
    main()
