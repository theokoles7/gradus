"""# Is Measured Forgetting Sensitive to Normalization Context?

Evaluate existing checkpoints only. No optimizer updates or new training.
Keep training-only calibration IDs, mixed order, views, and batch size fixed.
Compare both within-protocol mastery and the common initial-mastered cohort.

Save beside scratch_curriculum_feedback.py and scratch_curriculum_bn_audit.py.
    python scratch_curriculum_context_audit.py --device cuda
"""

from argparse       import ArgumentParser
from datetime       import datetime, timezone
from json           import dump, load
from pathlib        import Path
from time           import perf_counter
from types          import SimpleNamespace

import numpy        as np
import torch
from torchvision.datasets import CIFAR100

import scratch_curriculum_feedback as experiment
import scratch_curriculum_bn_audit as replay

# ARGUMENTS ========================================================================================

def parse_arguments():
    parser = ArgumentParser(description = "Checkpoint-only audit of curriculum forgetting measurements.")
    parser.add_argument("--source",             type = str, default = "analysis/curriculum_feedback")
    parser.add_argument("--device",             type = str, default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--seeds",              type = int, default = [1, 2], nargs = "+")
    parser.add_argument("--gaps",               type = int, default = [0, 1, 5, 10], nargs = "+")
    parser.add_argument("--calibration-images", type = int, default = 2048)
    parser.add_argument("--output",             type = str, default = "analysis/curriculum_context_audit")
    parser.add_argument("--smoke",              action = "store_true")
    args = parser.parse_args()
    args.gaps = sorted(set(args.gaps))
    if not args.gaps or args.gaps[0] != 0: parser.error("Include gap 0 for a valid initial-state comparison.")
    if args.calibration_images < 1: parser.error("Calibration size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA is unavailable.")
    return args

# COHORT MEASUREMENTS ===============================================================================

def collect(model, loaders, validation, device, cost):
    records = {name: experiment.score(model, loader, device, cost) for name, loader in loaders.items()}
    return records, experiment.validation_loss(model, validation, device, cost)


def calibrated_collect(model, loaders, validation, calibration, device, cost):
    normalization = [layer for layer in model.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [(layer, {key: value.detach().clone() for key, value in layer.named_buffers(recurse = False)}, layer.momentum)
             for layer in normalization]
    try:
        model.eval()
        for layer in normalization:
            layer.reset_running_stats(); layer.momentum = None; layer.train()
        with torch.no_grad():
            for x, _ in calibration:
                model(x.to(device, non_blocking = True)); cost["calibration_images"] += len(x)
        return collect(model, loaders, validation, device, cost)
    finally:
        for layer, buffers, momentum in saved:
            for name, value in buffers.items(): getattr(layer, name).copy_(value)
            layer.momentum = momentum
        model.eval()


def forgetting(correct, initial):
    return float((~correct[initial]).mean()) if initial.any() else float("nan")

# MAIN =============================================================================================

def main():
    from gradus.registration import NETWORK_REGISTRY

    args = parse_arguments()
    source = replay.locate_source(args)
    with open(source / "metadata.json") as handle: source_meta = load(handle)
    if bool(source_meta["arguments"]["smoke"]) != bool(args.smoke):
        raise ValueError("Source smoke flag and --smoke must agree.")
    if args.smoke:
        raw = SimpleNamespace(data = np.random.default_rng(1).integers(0, 256, size = (128, 32, 32, 3), dtype = np.uint8),
                              targets = np.repeat(np.arange(4), 32).tolist())
    else:
        raw = CIFAR100(root = source_meta["arguments"]["root"], train = True, download = False)
    labels = np.asarray(raw.targets)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output) / stamp; output.mkdir(parents = True, exist_ok = False)
    rows = []
    cost = {"audit_images": 0, "validation_images": 0, "calibration_images": 0}
    started = perf_counter()
    for seed in args.seeds:
        original = experiment.Namespace(**source_meta["arguments"]); original.device = args.device
        with np.load(source / f"cohorts_seed-{seed}.npz", allow_pickle = False) as identities:
            train = identities["train_ids"].copy()
            validation_ids = identities["validation_ids"].copy()
            masses = {"excluded": len(identities["pairs"]) / len(train),
                      "sentinel": len(identities["pairs"]) / len(train)}
            masses["remainder"] = 1 - masses["excluded"] - masses["sentinel"]
            panels = {name: identities[f"panel_{name}"].copy() for name in masses}
        if not set(train).isdisjoint(validation_ids): raise RuntimeError("Training/validation overlap.")
        rng = np.random.default_rng(seed + 61_000)
        calibration_ids = rng.permutation(experiment.stratified_subset(train, labels, args.calibration_images, rng))
        calibration_ids = calibration_ids[:len(calibration_ids) // original.batch_size * original.batch_size]
        if not len(calibration_ids): raise ValueError("Calibration requires a full training-size batch.")
        calibration = experiment.DataLoader(
            experiment.Images(raw, calibration_ids, seed), batch_size = original.batch_size,
            shuffle = False, num_workers = original.workers, pin_memory = args.device == "cuda",
            generator = torch.Generator().manual_seed(seed + 64_000)
        )
        loaders = {name: experiment.make_loader(raw, ids, seed, original) for name, ids in panels.items()}
        validation = experiment.make_loader(raw, validation_ids, seed, original)
        model = NETWORK_REGISTRY.load_network(network_id = "resnet-18", input_shape = (3, 32, 32),
                                              num_classes = len(np.unique(labels))).to(device)
        baseline, previous, archive = {}, {}, {}
        for gap in args.gaps:
            state = torch.load(source / f"checkpoint_seed-{seed}_gap-{gap}.pt", map_location = "cpu", weights_only = True)
            model.load_state_dict(state["model"]); del state
            native = collect(model, loaders, validation, device, cost)
            calibrated = calibrated_collect(model, loaders, validation, calibration, device, cost)
            protocols = {"native": native, "recalibrated": calibrated}
            if gap == 0: baseline = {name: value[0] for name, value in protocols.items()}
            for protocol, (records, loss) in protocols.items():
                row = {"seed": seed, "gap": gap, "bn_protocol": protocol, "validation_loss": loss}
                for name, record in records.items():
                    initial = baseline[protocol][name]["correct"].astype(bool)
                    common = baseline["native"][name]["correct"].astype(bool) & baseline["recalibrated"][name]["correct"].astype(bool)
                    row[f"{name}_loss"] = float(record["loss"].mean())
                    row[f"{name}_accuracy"] = float(record["correct"].mean())
                    row[f"{name}_within_protocol_forgetting"] = forgetting(record["correct"], initial)
                    row[f"{name}_common_mastered_forgetting"] = forgetting(record["correct"], common)
                    row[f"{name}_initial_correct_count"] = int(initial.sum())
                    row[f"{name}_common_mastered_count"] = int(common.sum())
                    for key, value in record.items(): archive[f"{protocol}_gap-{gap}_{name}_{key}"] = value
                row["paired_excluded_minus_sentinel_loss_change"] = float(
                    ((records["excluded"]["loss"] - baseline[protocol]["excluded"]["loss"]) -
                     (records["sentinel"]["loss"] - baseline[protocol]["sentinel"]["loss"])).mean())
                row["population_loss"] = sum(masses[name] * row[f"{name}_loss"] for name in masses)
                row["cached_population_loss"] = sum(masses[name] * (float(baseline[protocol][name]["loss"].mean())
                                                                   if name == "excluded" else row[f"{name}_loss"])
                                                    for name in masses)
                row["previous_observed_gap"] = previous[protocol]["gap"] if protocol in previous else 0
                for key in ("population_loss", "cached_population_loss"):
                    row[f"{key}_change"] = row[key] - previous[protocol][key] if protocol in previous else 0.0
                previous[protocol] = row; rows.append(row)
            experiment.write_csv(output / "context_audit.csv", rows)
            print(f"[seed {seed}, gap {gap}] native/recalibrated cohort audit complete", flush = True)
        np.savez_compressed(output / f"audit_arrays_seed-{seed}.npz", calibration_ids = calibration_ids, **archive)
        del model
    if args.device == "cuda": torch.cuda.synchronize()
    cost["wall_minutes"] = (perf_counter() - started) / 60
    lines = ["CURRICULUM FORGETTING: CHECKPOINT-ONLY NORMALIZATION AUDIT", "",
             f"Source: {source}; synthetic smoke: {args.smoke}; wall minutes: {cost['wall_minutes']:.3f}",
             "No optimizer updates. Fixed train-only calibration context across checkpoints.",
             "Forgetting = correct at gap0, wrong now; common-mastered denominator is shared across protocols.",
             "Changes are over observed intervals, not unobserved per-epoch transitions.", "",
             "seed gap protocol       val_CE   paired_loss_change  excluded_forgetting  common_excluded_forgetting"]
    for row in rows:
        lines.append(f"{row['seed']:4} {row['gap']:3} {row['bn_protocol']:13} {row['validation_loss']:8.4f} "
                     f"{row['paired_excluded_minus_sentinel_loss_change']:20.5f} {row['excluded_within_protocol_forgetting']:20.5f} "
                     f"{row['excluded_common_mastered_forgetting']:27.5f}")
    lines += ["", "Recalibration changes the evaluation protocol; it is not a new curriculum or proof that original forgetting was unreal.",
              "This validates the measurement before closing the hypothesis; it does not establish useful reentry.",
              f"Output: {output}"]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    with open(output / "metadata.json", "w") as handle:
        dump({"arguments": vars(args), "source": str(source), "costs": cost, "torch": torch.__version__}, handle, indent = 2)
    print(summary, flush = True)


if __name__ == "__main__":
    main()
