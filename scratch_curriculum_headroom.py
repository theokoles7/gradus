"""# Is There Actionable Curriculum Headroom Beyond Known Scores?

Reuse saved checkpoints. Each candidate first batch is followed by the SAME
ordinary shuffled-training suffix. Evaluate on two disjoint, previously unused
development panels. Choose on A, assess on B, and swap roles descriptively.

All endpoints use the fixed train-only BN recalibration protocol established
by the preceding audit. No test data or new training prefixes are used.

This is an expensive diagnostic selector, NOT a proposed efficient algorithm.
Gradient alignment is a simple comparator, not a reproduction of GLISTER.

Save beside scratch_curriculum_feedback.py and scratch_curriculum_bn_audit.py.
    python scratch_curriculum_headroom.py --device cuda
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
from torch.nn.functional import cross_entropy, softmax
from torchvision.datasets import CIFAR100

import scratch_curriculum_feedback as experiment
import scratch_curriculum_bn_audit as replay

# ARGUMENTS ========================================================================================

def parse_arguments():
    parser = ArgumentParser(description = "Cross-panel curriculum action-selection diagnostic.")
    parser.add_argument("--source",             type = str, default = "analysis/curriculum_feedback")
    parser.add_argument("--device",             type = str, default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--seeds",              type = int, default = [1, 2], nargs = "+")
    parser.add_argument("--gap",                type = int, default = 10)
    parser.add_argument("--candidates",         type = int, default = 64)
    parser.add_argument("--horizon",            type = int, default = 8)
    parser.add_argument("--panel-size",         type = int, default = 1000)
    parser.add_argument("--calibration-images", type = int, default = 2048)
    parser.add_argument("--output",             type = str, default = "analysis/curriculum_headroom")
    parser.add_argument("--smoke",              action = "store_true")
    args = parser.parse_args()
    args.gaps = [args.gap]                 # Interface to the existing checkpoint locator.
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA is unavailable.")
    if min(args.candidates, args.horizon, args.panel_size, args.calibration_images) < 1:
        parser.error("Counts must be positive.")
    if args.candidates < 2: parser.error("Use at least two candidate actions.")
    return args

# FIXED NORMALIZATION AND VALIDATION GRADIENTS ======================================================

@torch.no_grad()
def recalibrate(model, loader, device, cost):
    model.eval()
    normalization = [layer for layer in model.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [layer.momentum for layer in normalization]
    try:
        for layer in normalization:
            layer.reset_running_stats(); layer.momentum = None; layer.train()
        for x, _ in loader:
            model(x.to(device, non_blocking = True)); cost["calibration_images"] += len(x)
    finally:
        for layer, momentum in zip(normalization, momenta): layer.momentum = momentum
        model.eval()


def gradient_vector(model):
    return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters() if p.requires_grad])


def validation_gradient(model, loader, device, cost):
    """# First-Order Comparator: Treat the Calibrated Running Buffers as Fixed."""
    model.eval(); model.zero_grad(set_to_none = True)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        (cross_entropy(model(x), y, reduction = "sum") / len(loader.dataset)).backward()
        cost["validation_gradient_images"] += len(y)
    gradient = gradient_vector(model).clone()
    if not bool(torch.isfinite(gradient).all()): raise RuntimeError("Nonfinite validation reference gradient.")
    return gradient


def endpoints(model, validation, calibration, device, cost):
    """# Restore Endpoint Buffers Before Any Subsequent Training Update."""
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    try:
        recalibrate(model, calibration, device, cost)
        return {name: experiment.validation_loss(model, loader, device, cost)
                for name, loader in validation.items()}
    finally:
        for name, value in model.named_buffers(): value.copy_(buffers[name])
        model.eval()

# CANDIDATES AND PANELS ===============================================================================

def build_panels(raw, train, previously_used, seed, args):
    """# Reserve Two Fresh Panels from the Prospective Holdout, Not the Training Set."""
    labels = np.asarray(raw.targets)
    unavailable = np.zeros(len(labels), dtype = bool)
    unavailable[train] = True; unavailable[previously_used] = True
    available = np.flatnonzero(~unavailable)
    classes = np.unique(labels)
    if args.panel_size % len(classes): raise ValueError("Panel size must be divisible by the number of classes.")
    per_class = args.panel_size // len(classes)
    rng = np.random.default_rng(seed + 72_000)
    a, b = [], []
    for label in classes:
        pool = rng.permutation(available[labels[available] == label])
        if len(pool) < 2 * per_class:
            raise ValueError("Not enough previously unused validation examples; reduce --panel-size.")
        a.extend(pool[:per_class]); b.extend(pool[per_class:2 * per_class])
    return {"A": np.asarray(a), "B": np.asarray(b)}


def class_matched_candidates(train, labels, batch_size, count, rng):
    template = labels[rng.choice(train, batch_size, replace = False)]
    candidates = []
    for _ in range(count):
        indices = np.empty(batch_size, dtype = np.int64)
        for label in np.unique(template):
            positions = np.flatnonzero(template == label)
            pool = train[labels[train] == label]
            indices[positions] = rng.choice(pool, len(positions), replace = False)
        candidates.append(indices)
    return candidates

# HEADROOM MEASUREMENTS ===============================================================================

def run_seed(raw, source, seed, args, output):
    from gradus.registration import NETWORK_REGISTRY

    device = torch.device(args.device)
    state = torch.load(source / f"checkpoint_seed-{seed}_gap-{args.gap}.pt", map_location = "cpu", weights_only = True)
    original = experiment.Namespace(**state["arguments"]); original.device = args.device
    if bool(original.smoke) != bool(args.smoke): raise ValueError("Source smoke flag and --smoke disagree.")
    with np.load(source / f"cohorts_seed-{seed}.npz", allow_pickle = False) as identities:
        train = identities["train_ids"].copy(); used_validation = identities["validation_ids"].copy()
    labels = np.asarray(raw.targets)
    panel_ids = build_panels(raw, train, used_validation, seed, args)
    validation = {name: experiment.make_loader(raw, ids, seed, original) for name, ids in panel_ids.items()}
    calibration_rng = np.random.default_rng(seed + 61_000)
    calibration_ids = calibration_rng.permutation(experiment.stratified_subset(train, labels, args.calibration_images, calibration_rng))
    calibration_ids = calibration_ids[:len(calibration_ids) // original.batch_size * original.batch_size]
    if not len(calibration_ids): raise ValueError("Calibration requires a full training-size batch.")
    calibration = experiment.DataLoader(
        experiment.Images(raw, calibration_ids, seed), batch_size = original.batch_size,
        shuffle = False, num_workers = original.workers, pin_memory = args.device == "cuda",
        generator = torch.Generator().manual_seed(seed + 64_000)
    )
    model = NETWORK_REGISTRY.load_network(network_id = "resnet-18", input_shape = (3, 32, 32),
                                          num_classes = len(np.unique(labels))).to(device)
    optimizer = experiment.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    experiment.restore(model, optimizer, state)
    cost = {key: 0 for key in ("calibration_images", "validation_images", "validation_gradient_images", "branch_images", "branch_updates")}
    recalibrate(model, calibration, device, cost)
    starting = {name: experiment.validation_loss(model, loader, device, cost) for name, loader in validation.items()}
    gradients = {name: validation_gradient(model, loader, device, cost) for name, loader in validation.items()}
    # Reference gradients are only scored, never applied to training parameters.
    experiment.restore(model, optimizer, state)
    rng = np.random.default_rng(seed + 75_000)
    candidates = class_matched_candidates(train, labels, original.batch_size, args.candidates, rng)
    suffix = [rng.choice(train, original.batch_size, replace = False) for _ in range(args.horizon - 1)]
    key = 2_000_000 + args.gap * 10_000
    suffix_batches = [experiment.batch_tensors(raw, ids, seed, key + step + 1, device)
                      for step, ids in enumerate(suffix)]
    np.savez_compressed(output / f"identities_seed-{seed}.npz", panel_A = panel_ids["A"], panel_B = panel_ids["B"],
                        calibration = calibration_ids, candidates = np.stack(candidates),
                        suffix = np.asarray(suffix, dtype = np.int64))
    rows = []
    for candidate, ids in enumerate(candidates):
        experiment.restore(model, optimizer, state)
        # A common key makes shared examples receive identical first-action views.
        first = experiment.batch_tensors(raw, ids, seed, key, device)
        descriptors = {}
        for step in range(args.horizon):
            x, y = first if step == 0 else suffix_batches[step - 1]
            model.train(); optimizer.zero_grad(set_to_none = True)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            if step == 0:
                with torch.no_grad():
                    probability = softmax(logits.detach(), dim = 1)
                    probability[torch.arange(len(y), device = device), y] -= 1
                    vector = gradient_vector(model)
                    descriptors = {"batch_loss": float(loss.detach()), "batch_el2n": float(probability.norm(dim = 1).mean()),
                                   "gradient_norm": float(vector.norm()),
                                   **{f"alignment_{name}": float(torch.dot(vector, gradient)) for name, gradient in gradients.items()}}
                    if not all(np.isfinite(value) for value in descriptors.values()):
                        raise RuntimeError(f"Nonfinite descriptor for candidate {candidate}; do not silently discard candidates.")
                    del vector
            optimizer.step()
            cost["branch_images"] += len(y); cost["branch_updates"] += 1
            if step == 0 or step == args.horizon - 1:
                values = endpoints(model, validation, calibration, device, cost)
                if not all(np.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"Nonfinite endpoint for candidate {candidate}.")
                rows.append({"seed": seed, "gap": args.gap, "candidate": candidate, "horizon": step + 1,
                             **descriptors, **{f"loss_{name}": value for name, value in values.items()},
                             **{f"starting_loss_{name}": value for name, value in starting.items()}})
        experiment.write_csv(output / f"candidate_values_seed-{seed}.csv", rows)
        print(f"[seed {seed}] candidate {candidate + 1}/{args.candidates} complete", flush = True)
    return rows, cost

# CROSS-PANEL SELECTION ==============================================================================

def assess(rows):
    frame = pd.DataFrame(rows)
    selected, agreement = [], []
    for (seed, horizon), group in frame.groupby(["seed", "horizon"]):
        group = group.reset_index(drop = True)
        agreement.append({"seed": int(seed), "horizon": int(horizon),
                          "candidate_rank_correlation_A_B": float(group.loss_A.corr(group.loss_B, method = "spearman"))})
        for choose, evaluate in (("A", "B"), ("B", "A")):
            selectors = {"lookahead": int(group[f"loss_{choose}"].argmin()),
                         "gradient_alignment": int(group[f"alignment_{choose}"].argmax()),
                         "high_loss": int(group.batch_loss.argmax()), "low_loss": int(group.batch_loss.argmin()),
                         "high_el2n": int(group.batch_el2n.argmax()), "low_el2n": int(group.batch_el2n.argmin()),
                         "high_gradient_norm": int(group.gradient_norm.argmax())}
            for method, index in selectors.items():
                selected.append({"seed": int(seed), "horizon": int(horizon), "choose_panel": choose,
                                 "evaluate_panel": evaluate, "selector": method,
                                 "selected_candidate": int(group.iloc[index].candidate),
                                 "gain_on_selection_panel": float(group[f"loss_{choose}"].mean() - group.iloc[index][f"loss_{choose}"]),
                                 "gain_on_other_panel": float(group[f"loss_{evaluate}"].mean() - group.iloc[index][f"loss_{evaluate}"])})
    return selected, agreement


def main():
    args = parse_arguments()
    source = replay.locate_source(args)
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
    print(f"Output directory: {output}", flush = True)
    all_rows, costs = [], {}
    started = perf_counter()
    for seed in args.seeds:
        rows, cost = run_seed(raw, source, seed, args, output)
        all_rows.extend(rows); costs[str(seed)] = cost
        selections, agreement = assess(all_rows)
        experiment.write_csv(output / "cross_panel_selection.csv", selections)
        experiment.write_csv(output / "panel_agreement.csv", agreement)
        experiment.write_csv(output / "candidate_values.csv", all_rows)
    if args.device == "cuda": torch.cuda.synchronize()
    lines = ["CURRICULUM ACTION-SELECTION HEADROOM: EXPLORATORY DEVELOPMENT DIAGNOSTIC", "",
             f"Source: {source}; synthetic smoke: {args.smoke}; wall minutes: {(perf_counter() - started) / 60:.3f}",
             "Primary horizon is the prespecified longer horizon. H1 is a secondary diagnostic.",
             "A and B are disjoint, previously unused development panels. Neither is the official test set.",
             "All candidates share the class histogram, starting state, suffix updates and endpoint protocol.",
             "The suffix samples from the whole training split, not only the earlier active pool.",
             "Positive gains are lower CE than uniformly choosing one candidate from this fixed pool.", "",
             "seed H choose->evaluate selector                  other_panel_gain selection_panel_gain"]
    for row in selections:
        lines.append(f"{row['seed']:4} {row['horizon']:2} {row['choose_panel']}->{row['evaluate_panel']:13} "
                     f"{row['selector']:25} {row['gain_on_other_panel']:16.6f} {row['gain_on_selection_panel']:20.6f}")
    lines += ["", "CANDIDATE RANK AGREEMENT"]
    for row in agreement: lines.append(str(row))
    lines += ["", "Lookahead is an expensive diagnostic search, not a proposed efficient or novel algorithm.",
              "The alignment comparator holds calibrated reference buffers fixed in its gradient; it is not an exact GLISTER reproduction.",
              "A residual beyond this comparator is not proof of novelty or proof that history caused utility.",
              "A/B directions share candidates and models; they are dependent, not extra training seeds.",
              "Lookahead knows one fixed future suffix; panel transfer does not establish transfer to other future training sequences.",
              "Results concern these saved post-withholding states, not all stages of conventional SGD.",
              "Candidate search, validation gradients, calibration and branch costs are all recorded in metadata.json.",
              "Do not select a favorable horizon, panel direction or seed after inspecting results.",
              f"Output: {output}"]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    with open(output / "metadata.json", "w") as handle:
        dump({"arguments": vars(args), "source": str(source), "costs": costs, "torch": torch.__version__}, handle, indent = 2)
    print(summary, flush = True)


if __name__ == "__main__":
    main()
