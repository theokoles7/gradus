"""# Interaction-Aware Curriculum: First Ordering Falsification

Reuse saved ResNet checkpoints. Evaluate every order of the SAME four batches.
Compare a full-momentum second-order planner with random order, first-order
momentum-aware order, frozen-gradient geometry, and scalar priorities.

Finite-difference Hessian actions avoid double-backward GPU memory overhead.
The first window also checks derivative-table sensitivity at half the FD step.
This is a local mechanism test, not a novel-algorithm or final-accuracy claim.

Save beside scratch_curriculum_headroom.py and its supporting scripts.
    python scratch_curriculum_window_order.py --device cuda
"""

from argparse       import ArgumentParser
from datetime       import datetime, timezone
from itertools      import permutations
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

# ARGUMENTS ========================================================================================

def parse_arguments():
    parser = ArgumentParser(description = "Matched-exposure local curriculum-order experiment.")
    parser.add_argument("--headroom",           type = str,   default = "analysis/curriculum_headroom")
    parser.add_argument("--source",             type = str,   default = None)
    parser.add_argument("--device",             type = str,   default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--seeds",              type = int,   default = [1, 2], nargs = "+")
    parser.add_argument("--windows",            type = int,   default = 4)
    parser.add_argument("--buffer",             type = int,   default = 4, choices = [2, 3, 4])
    parser.add_argument("--planning-size",      type = int,   default = 200)
    parser.add_argument("--fd-relative",        type = float, default = 1e-4)
    parser.add_argument("--output",             type = str,   default = "analysis/curriculum_window_order")
    parser.add_argument("--smoke",              action = "store_true")
    args = parser.parse_args()
    if min(args.windows, args.planning_size) < 1 or not np.isfinite(args.fd_relative) or args.fd_relative <= 0:
        parser.error("Counts and finite-difference step must be positive and finite.")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA is unavailable.")
    return args


def locate_headroom(args):
    root = Path(args.headroom)
    candidates = [root] if (root / "metadata.json").exists() else list(root.glob("*"))
    valid = [p for p in candidates if (p / "metadata.json").exists()
             and all((p / f"identities_seed-{s}.npz").exists() for s in args.seeds)]
    if not valid: raise FileNotFoundError("No completed headroom run found; provide its exact --headroom path.")
    return max(valid, key = lambda p: (p / "metadata.json").stat().st_mtime)

# PARAMETERS, BUFFERS AND FINITE DIFFERENCES =========================================================

def flat_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def buffers_of(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


@torch.no_grad()
def set_point(model, point, buffers):
    position = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            count = parameter.numel()
            parameter.copy_(point[position:position + count].reshape_as(parameter))
            position += count
    for name, value in model.named_buffers(): value.copy_(buffers[name])


def batch_gradient(model, point, buffers, batch, decay, cost):
    set_point(model, point, buffers)
    model.train(); model.zero_grad(set_to_none = True)
    x, y = batch
    loss = cross_entropy(model(x), y)
    loss.backward()
    gradient = headroom.gradient_vector(model)
    cost["planning_train_images"] += len(y)
    return gradient + decay * point, float(loss.detach()), float(gradient.norm())


def validation_gradient(model, point, buffers, loader, device, cost):
    set_point(model, point, buffers)
    model.eval(); model.zero_grad(set_to_none = True)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        (cross_entropy(model(x), y, reduction = "sum") / len(loader.dataset)).backward()
        cost["planning_validation_images"] += len(y)
    return headroom.gradient_vector(model)


def finite_difference(gradient_function, point, direction, relative):
    norm = float(direction.norm())
    if norm == 0: return torch.zeros_like(point)
    # All directions use the same PARAMETER perturbation norm, not the same coefficient step.
    step = relative * max(1.0, float(point.norm())) / norm
    plus = gradient_function(point + step * direction)
    minus = gradient_function(point - step * direction)
    result = (plus - minus) / (2 * step)
    if not bool(torch.isfinite(result).all()): raise RuntimeError("Nonfinite finite-difference curvature estimate.")
    return result


def derivative_tables(model, theta, reference, batches, basis, u, planning, device, decay, relative, cost):
    interaction = []
    for batch in batches:
        def gradient(point): return batch_gradient(model, point, reference, batch, decay, cost)[0]
        hu = finite_difference(gradient, theta, u, relative)
        interaction.append(basis @ hu)
    columns = []
    for direction in basis:
        def gradient(point): return validation_gradient(model, point, reference, planning, device, cost)
        hv = finite_difference(gradient, theta, direction, relative)
        columns.append(basis @ hv)
    m = torch.stack(interaction).detach().double().cpu().numpy()
    raw_q = torch.stack(columns, dim = 1).detach().double().cpu().numpy()
    # Only the symmetric part contributes to p^T Q p. Record asymmetry as a numerical diagnostic.
    asymmetry = float(np.linalg.norm(raw_q - raw_q.T) / max(np.linalg.norm(raw_q), 1e-30))
    return m, (raw_q + raw_q.T) / 2, asymmetry

# SMALL COEFFICIENT-SPACE PLANNER ====================================================================

def score_order(order, s, m, q, momentum, learning_rate):
    r = np.zeros(len(s)); r[0] = 1
    p = np.zeros(len(s))
    alpha = beta = 0.0
    for batch in order:
        alpha = momentum * alpha + m[batch] @ p
        beta -= alpha
        r = momentum * r; r[batch + 1] += 1
        p -= r
    first = learning_rate * (s @ p)
    geometry = first + learning_rate**2 * (0.5 * p @ q @ p)
    second = geometry + learning_rate**2 * beta
    return float(first), float(geometry), float(second)

# FROZEN-REFERENCE AND POPULATION-CALIBRATED ENDPOINTS ================================================

def measure(model, reference, planning, assessment, calibration, device, cost):
    actual_buffers = buffers_of(model)
    try:
        with torch.no_grad():
            for name, value in model.named_buffers(): value.copy_(reference[name])
        a = experiment.validation_loss(model, planning, device, cost)
        b = experiment.validation_loss(model, assessment, device, cost)
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers(): value.copy_(actual_buffers[name])
    calibrated = headroom.endpoints(model, {"B": assessment}, calibration, device, cost)["B"]
    return {"loss_A_fixed": a, "loss_B_fixed": b, "loss_B_calibrated": calibrated}

# CHECKPOINT EXPERIMENT ===============================================================================

def run_seed(raw, source, run, seed, gap, args, output):
    from gradus.registration import NETWORK_REGISTRY

    device = torch.device(args.device)
    state = torch.load(source / f"checkpoint_seed-{seed}_gap-{gap}.pt", map_location = "cpu", weights_only = True)
    original = experiment.Namespace(**state["arguments"]); original.device = args.device
    if bool(original.smoke) != bool(args.smoke): raise ValueError("Source smoke flag and --smoke disagree.")
    with np.load(source / f"cohorts_seed-{seed}.npz", allow_pickle = False) as data: train = data["train_ids"].copy()
    with np.load(run / f"identities_seed-{seed}.npz", allow_pickle = False) as data:
        panel_a, panel_b, calibration_ids = data["panel_A"].copy(), data["panel_B"].copy(), data["calibration"].copy()
    labels = np.asarray(raw.targets)
    if set(panel_a).intersection(train) or set(panel_b).intersection(train) or set(panel_a).intersection(panel_b):
        raise RuntimeError("Training/development panel overlap.")
    planning_ids = experiment.stratified_subset(panel_a, labels, min(args.planning_size, len(panel_a)), np.random.default_rng(seed + 93_000))
    planning = experiment.make_loader(raw, planning_ids, seed, original)
    assessment = experiment.make_loader(raw, panel_b, seed, original)
    calibration = experiment.DataLoader(
        experiment.Images(raw, calibration_ids, seed), batch_size = original.batch_size,
        shuffle = False, num_workers = original.workers, pin_memory = args.device == "cuda",
        generator = torch.Generator().manual_seed(seed + 64_000)
    )
    model = NETWORK_REGISTRY.load_network(network_id = "resnet-18", input_shape = (3, 32, 32),
                                          num_classes = len(np.unique(labels))).to(device)
    optimizer = experiment.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    experiment.restore(model, optimizer, state)
    if len(optimizer.param_groups) != 1: raise ValueError("This diagnostic expects one SGD parameter group.")
    group = optimizer.param_groups[0]
    if group.get("dampening", 0) or group.get("nesterov", False) or group.get("maximize", False):
        raise ValueError("The surrogate assumes zero dampening, no Nesterov and minimization.")
    learning_rate, momentum, decay = float(group["lr"]), float(group["momentum"]), float(group["weight_decay"])
    cost = {key: 0 for key in ("planning_train_images", "planning_validation_images", "calibration_images",
                               "validation_images", "branch_images", "branch_updates")}
    theta = flat_parameters(model)
    velocity = torch.cat([optimizer.state[p].get("momentum_buffer", torch.zeros_like(p)).reshape(-1)
                          for p in model.parameters() if p.requires_grad]).detach().clone()
    headroom.recalibrate(model, calibration, device, cost)
    reference = buffers_of(model)
    u = validation_gradient(model, theta, reference, planning, device, cost)
    all_orders = list(permutations(range(args.buffer)))
    rows, selected_rows, stability_rows = [], [], []
    for window in range(args.windows):
        rng = np.random.default_rng(seed * 1_000_003 + 950_000 + window)
        ids = rng.choice(train, args.buffer * original.batch_size, replace = False).reshape(args.buffer, original.batch_size)
        batches = [experiment.batch_tensors(raw, batch, seed, 4_000_000 + window * 100 + i, device)
                   for i, batch in enumerate(ids)]
        np.save(output / f"batches_seed-{seed}_window-{window}.npy", ids)
        gradients, losses, norms = [], [], []
        for batch in batches:
            g, loss, norm = batch_gradient(model, theta, reference, batch, decay, cost)
            if not bool(torch.isfinite(g).all()): raise RuntimeError("Nonfinite planning gradient.")
            gradients.append(g); losses.append(loss); norms.append(norm)
        basis = torch.stack([velocity] + gradients)
        s = (basis @ u).detach().double().cpu().numpy()
        m, q, asymmetry = derivative_tables(model, theta, reference, batches, basis, u, planning, device,
                                          decay, args.fd_relative, cost)
        scores = np.asarray([score_order(order, s, m, q, momentum, learning_rate) for order in all_orders])
        if not np.isfinite(scores).all(): raise RuntimeError("Nonfinite planning surrogate.")
        selected = {"first_order": int(scores[:, 0].argmin()), "frozen_gradient_geometry": int(scores[:, 1].argmin()),
                    "second_order": int(scores[:, 2].argmin()),
                    "high_gradient_norm": all_orders.index(tuple(np.argsort(-np.asarray(norms), kind = "stable"))),
                    "high_loss": all_orders.index(tuple(np.argsort(-np.asarray(losses), kind = "stable")))}
        decay_matrix = np.repeat((decay * s)[None, :], args.buffer, axis = 0)
        decay_scores = [score_order(order, s, decay_matrix, q, momentum, learning_rate)[2] for order in all_orders]
        selected["decay_only_geometry"] = int(np.argmin(decay_scores))
        if window == 0:
            m_half, q_half, asym_half = derivative_tables(model, theta, reference, batches, basis, u, planning,
                                                        device, decay, args.fd_relative / 2, cost)
            half_all = np.asarray([score_order(order, s, m_half, q_half, momentum, learning_rate) for order in all_orders])
            half_scores = half_all[:, 2]
            quadratic = scores[:, 2] - scores[:, 0]; quadratic -= quadratic.mean()
            quadratic_half = half_all[:, 2] - half_all[:, 0]; quadratic_half -= quadratic_half.mean()
            interaction = scores[:, 2] - scores[:, 1]; interaction -= interaction.mean()
            interaction_half = half_all[:, 2] - half_all[:, 1]; interaction_half -= interaction_half.mean()
            stability_rows.append({"seed": seed, "window": window, "fd_relative": args.fd_relative,
                                   "parameter_perturbation_norm": args.fd_relative * max(1.0, float(theta.norm())),
                                   "relative_M_change_at_half_step": float(np.linalg.norm(m - m_half) / max(np.linalg.norm(m_half), 1e-30)),
                                   "relative_Q_change_at_half_step": float(np.linalg.norm(q - q_half) / max(np.linalg.norm(q_half), 1e-30)),
                                   "Q_asymmetry": asymmetry, "Q_half_step_asymmetry": asym_half,
                                   "relative_centered_quadratic_order_change": float(np.linalg.norm(quadratic - quadratic_half) / max(np.linalg.norm(quadratic_half), 1e-30)),
                                   "relative_centered_training_interaction_change": float(np.linalg.norm(interaction - interaction_half) / max(np.linalg.norm(interaction_half), 1e-30)),
                                   "same_chosen_order_at_half_step": bool(half_scores.argmin() == selected["second_order"])})
        del basis, gradients
        window_rows = []
        for order_index, order in enumerate(all_orders):
            experiment.restore(model, optimizer, state)
            for batch in order:
                x, y = batches[batch]
                model.train(); optimizer.zero_grad(set_to_none = True)
                cross_entropy(model(x), y).backward(); optimizer.step()
                cost["branch_images"] += len(y); cost["branch_updates"] += 1
            values = measure(model, reference, planning, assessment, calibration, device, cost)
            if not all(np.isfinite(value) for value in values.values()): raise RuntimeError("Nonfinite order endpoint.")
            window_rows.append({"seed": seed, "window": window, "order_index": order_index,
                                "order": "-".join(str(i) for i in order),
                                "first_order_score": scores[order_index, 0], "geometry_score": scores[order_index, 1],
                                "second_order_score": scores[order_index, 2], **values})
        frame = pd.DataFrame(window_rows)
        selected["actual_lookahead_A"] = int(frame.loss_A_fixed.argmin())
        for name, order_index in selected.items():
            chosen = frame.iloc[order_index]
            selected_rows.append({"seed": seed, "window": window, "policy": name, "order": chosen.order,
                                  "gain_A_fixed": float(frame.loss_A_fixed.mean() - chosen.loss_A_fixed),
                                  "gain_B_fixed": float(frame.loss_B_fixed.mean() - chosen.loss_B_fixed),
                                  "gain_B_calibrated": float(frame.loss_B_calibrated.mean() - chosen.loss_B_calibrated),
                                  "loss_B_fixed": float(chosen.loss_B_fixed), "loss_B_calibrated": float(chosen.loss_B_calibrated)})
        rows.extend(window_rows)
        experiment.write_csv(output / f"order_values_seed-{seed}.csv", rows)
        experiment.write_csv(output / f"selected_orders_seed-{seed}.csv", selected_rows)
        experiment.write_csv(output / f"fd_stability_seed-{seed}.csv", stability_rows)
        print(f"[seed {seed}] window {window + 1}/{args.windows}: all {len(all_orders)} orders complete", flush = True)
    return rows, selected_rows, stability_rows, cost

# MAIN ==============================================================================================

def main():
    args = parse_arguments()
    run = locate_headroom(args)
    with open(run / "metadata.json") as handle: previous = load(handle)
    args.source = args.source or previous["source"]
    args.gaps = [int(previous["arguments"]["gap"])]
    source = headroom.replay.locate_source(args)
    with open(source / "metadata.json") as handle: source_meta = load(handle)
    if args.smoke:
        raw = SimpleNamespace(data = np.random.default_rng(1).integers(0, 256, size = (128, 32, 32, 3), dtype = np.uint8),
                              targets = np.repeat(np.arange(4), 32).tolist())
    else:
        raw = CIFAR100(root = source_meta["arguments"]["root"], train = True, download = False)
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents = True, exist_ok = False)
    print(f"Headroom input: {run}\nOutput directory: {output}", flush = True)
    rows, selected, stability, costs = [], [], [], {}
    started = perf_counter()
    for seed in args.seeds:
        r, s, f, c = run_seed(raw, source, run, seed, args.gaps[0], args, output)
        rows.extend(r); selected.extend(s); stability.extend(f); costs[str(seed)] = c
        experiment.write_csv(output / "order_values.csv", rows)
        experiment.write_csv(output / "selected_orders.csv", selected)
        experiment.write_csv(output / "fd_stability.csv", stability)
    if args.device == "cuda": torch.cuda.synchronize()
    frame = pd.DataFrame(selected)
    lines = ["INTERACTION-AWARE WINDOW ORDERING: FIRST FALSIFICATION", "",
             f"Input: {run}; smoke: {args.smoke}; wall minutes: {(perf_counter() - started) / 60:.3f}",
             "All orders process exactly the same examples/views once from the same full optimizer state.",
             "Stored learning rate and momentum are preserved; no training prefix or official test evaluation.",
             "Planner differentiates a fixed-reference BN validation objective; B-fixed is its target-matched cross-panel check.",
             "B-calibrated is a separate practical sensitivity endpoint, not the same differentiated objective.",
             "FD tables are estimates on a ReLU network; inspect half-step stability before interpreting them as curvature.", "",
             "seed policy                         mean_B_fixed_gain mean_B_calibrated_gain"]
    for (seed, name), group in frame.groupby(["seed", "policy"]):
        lines.append(f"{seed:4} {name:30} {group.gain_B_fixed.mean():17.6f} {group.gain_B_calibrated.mean():22.6f}")
    lines += ["", "FINITE-DIFFERENCE STABILITY (first window of each seed)"]
    lines.extend(str(row) for row in stability)
    lines += ["", "The primary comparison is second_order vs first_order AND frozen_gradient_geometry AND decay_only_geometry.",
              "The decay-only control includes known weight-decay curvature but no data-loss curvature.",
              "Positive gain is lower CE than the exact mean over every order of the same window.",
              "Windows share trained states; they are not independent training seeds.",
              "Derivative acquisition and exhaustive evaluation are diagnostic costs, not an efficient deployed algorithm.",
              "No novelty, final-accuracy or long-training benefit is established by this run.",
              f"Output: {output}"]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    with open(output / "metadata.json", "w") as handle:
        dump({"arguments": vars(args), "source": str(source), "costs": costs, "torch": torch.__version__}, handle, indent = 2)
    print(summary, flush = True)


if __name__ == "__main__":
    main()
