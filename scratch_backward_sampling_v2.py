"""# Selective Backward Computation: Dense-Adjoint Control

Sample convolution weight gradients while keeping every input adjoint dense.
This removes propagation of sampling noise through depth. It is a mechanism
control, not a new algorithm or a training experiment.

Uses one fixed synthetic CIFAR-shaped batch. No datasets, downloads, training,
test-set evaluation, or changes to Gradus source files are required.

Run from the Gradus repository root after saving this file there:
    python scratch_backward_sampling_v2.py --device cuda --mode weight

Outputs: summary.txt, results.csv, metadata.json under the printed run directory.
This is a feasibility diagnostic, not a novelty or generalization experiment.
"""

from argparse       import ArgumentParser, Namespace
from copy           import deepcopy
from datetime       import datetime, timezone
from json           import dump
from math           import ceil, isfinite, sqrt
from pathlib        import Path
from statistics     import median
from time           import perf_counter
from typing         import Dict, List

import csv
import platform
import random

import torch
from torch          import Tensor
from torch.autograd import Function
from torch.nn       import Conv2d, Module
from torch.nn.functional import conv2d, cross_entropy

# ARGUMENTS ========================================================================================

def parse_arguments() -> Namespace:
    """# Parse Diagnostic Arguments."""
    parser = ArgumentParser(description = "Profile sampled convolution backward on frozen ResNet-18.")
    parser.add_argument("--mode",        type = str,   default = "weight", choices = ["weight", "joint"],
                        help = "weight: sample parameter gradients only; joint: reproduce original estimator.")
    parser.add_argument("--device",      type = str,   default = "cuda", choices = ["cpu", "cuda"])
    parser.add_argument("--fractions",   type = float, default = [0.5, 0.25], nargs = "+")
    parser.add_argument("--batch-size",  type = int,   default = 128)
    parser.add_argument("--trials",      type = int,   default = 64)
    parser.add_argument("--warmup",      type = int,   default = 10)
    parser.add_argument("--steps",       type = int,   default = 20)
    parser.add_argument("--repeats",     type = int,   default = 3)
    parser.add_argument("--seed",        type = int,   default = 1)
    parser.add_argument("--threads",     type = int,   default = 2)
    parser.add_argument("--checkpoint",  type = str,   default = None)
    parser.add_argument("--output",      type = str,   default = "analysis/backward_sampling_v2")
    args = parser.parse_args()
    if args.batch_size < 2 or args.trials < 2:
        parser.error("--batch-size and --trials must both be >= 2.")
    if min(args.steps, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("Steps, repeats and threads must be positive; warmup must be nonnegative.")
    if not args.fractions or any(not 0 < q < 1 for q in args.fractions):
        parser.error("Sampling fractions must be strictly between zero and one.")
    if len(set(args.fractions)) != len(args.fractions):
        parser.error("Do not repeat sampling fractions.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu only for a small smoke check.")
    return args

# SAMPLED CONVOLUTION ===============================================================================

class SampledConvolution(Function):
    """# Full Forward; Fresh Example Sampling for Each Convolution Reverse Operation."""

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, fraction, mode):
        ctx.save_for_backward(x, weight)
        ctx.configuration = (stride, padding, dilation, groups)
        ctx.fraction = fraction
        ctx.mode = mode
        return conv2d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, incoming):
        x, weight = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.configuration
        n = x.shape[0]
        k = max(1, ceil(ctx.fraction * n))

        # A fresh fixed-size subset gives inclusion probability k / n.
        indices = torch.randperm(n, device = x.device)[:k] if k < n else None
        selected_x = x.index_select(0, indices) if indices is not None else x
        selected_d = incoming.index_select(0, indices) * (n / k) if indices is not None else incoming
        selected_x, selected_d = selected_x.contiguous(), selected_d.contiguous()

        # These kernels actually receive k examples, not a dense masked batch.
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            if ctx.mode == "weight":
                # The control: no random mask enters the propagated adjoint.
                dx = torch.nn.grad.conv2d_input(
                    x.shape, weight, incoming.contiguous(), stride, padding, dilation, groups
                )
            else:
                selected_dx = torch.nn.grad.conv2d_input(
                    selected_x.shape, weight, selected_d, stride, padding, dilation, groups
                )
                if indices is None:
                    dx = selected_dx
                else:
                    dx = torch.zeros_like(x)
                    dx.index_copy_(0, indices, selected_dx)
        if ctx.needs_input_grad[1]:
            dw = torch.nn.grad.conv2d_weight(
                selected_x, weight.shape, selected_d, stride, padding, dilation, groups
            )
        if ctx.needs_input_grad[2]:
            db = selected_d.sum(dim = (0, 2, 3))
        return dx, dw, db, None, None, None, None, None, None


class SampledConv2d(Module):
    """# Preserve Convolution Parameters and Full Forward Activations."""

    def __init__(self, original: Conv2d):
        super().__init__()
        if original.padding_mode != "zeros" or isinstance(original.padding, str):
            raise ValueError("This diagnostic only supports numeric, zero-padded convolutions.")
        self.weight = original.weight
        self.bias = original.bias
        self.configuration = (original.stride, original.padding, original.dilation, original.groups)
        self.fraction = 1.0
        self.mode = "weight"

    def forward(self, x: Tensor) -> Tensor:
        return SampledConvolution.apply(x, self.weight, self.bias, *self.configuration, self.fraction, self.mode)


def replace_convolutions(network: Module) -> None:
    """# Replace Only Convolutions; Keep Native BatchNorm and Residual Backpropagation."""
    for name, child in list(network.named_children()):
        if isinstance(child, Conv2d):
            setattr(network, name, SampledConv2d(child))
        else:
            replace_convolutions(child)


def set_fraction(network: Module, fraction: float) -> None:
    for layer in network.modules():
        if isinstance(layer, SampledConv2d):
            layer.fraction = fraction

# GRADIENT DIAGNOSTICS ===============================================================================

def backward_step(network: Module, x: Tensor, y: Tensor) -> Tensor:
    """# Forward/Backward Only: Parameters Stay Fixed; No Optimizer Step Is Taken."""
    network.zero_grad(set_to_none = True)
    logits = network(x)
    cross_entropy(logits, y).backward()
    return logits.detach()


def gradient_vector(network: Module) -> Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in network.parameters() if p.requires_grad])


def gradient_statistics(network: Module, x: Tensor, y: Tensor, reference: Tensor, trials: int) -> Dict:
    """# Online Mean/Trace Variance; No Trial Gradient Archive Is Required."""
    mean = torch.zeros_like(reference)
    m2 = torch.zeros((), device = reference.device, dtype = torch.float64)
    cosine_sum = torch.zeros((), device = reference.device, dtype = torch.float64)
    error_sum = torch.zeros_like(cosine_sum)
    norm = reference.norm().clamp_min(1e-30)
    for trial in range(1, trials + 1):
        backward_step(network, x, y)
        vector = gradient_vector(network)
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError(f"Nonfinite sampled gradient at draw {trial}; stop this configuration.")
        difference = vector - mean
        mean.add_(difference / trial)
        m2.add_(torch.dot(difference, vector - mean).double())
        error_sum.add_((vector - reference).square().sum().double())
        cosine_sum.add_((torch.dot(vector, reference) / (vector.norm().clamp_min(1e-30) * norm)).double())
        if trial == trials or trial % max(1, trials // 4) == 0:
            print(f"    gradient draws: {trial}/{trials}", flush = True)

    # MCSE norm is sqrt(trace(sample covariance) / trials), not a confidence interval.
    return {
        "relative_mean_error":      float((mean - reference).norm() / norm),
        "relative_mcse_norm":       sqrt(max(0.0, float(m2)) / (trials * (trials - 1))) / float(norm),
        "relative_single_draw_rms": sqrt(float(error_sum) / trials) / float(norm),
        "mean_gradient_cosine":     float(cosine_sum / trials),
    }

# SYNCHRONIZED WALL-TIME PROFILING ==================================================================

def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile(networks: Dict, x: Tensor, y: Tensor, args: Namespace) -> Dict:
    """# Time Native and Custom Paths Independently; Include Dispatch, Gather and Scatter."""
    measurements = {name: [] for name in networks}
    order = list(networks)
    rng = random.Random(args.seed + 10_000)
    for repeat in range(args.repeats):
        rng.shuffle(order)
        for name in order:
            network, fraction = networks[name]
            set_fraction(network, fraction)
            for _ in range(args.warmup):
                backward_step(network, x, y)
            synchronize(x.device)
            started = perf_counter()
            for _ in range(args.steps):
                backward_step(network, x, y)
            synchronize(x.device)
            elapsed = 1000 * (perf_counter() - started) / args.steps
            measurements[name].append(elapsed)
            print(f"    timing {repeat + 1}/{args.repeats}: {name:16} {elapsed:9.3f} ms", flush = True)
    return measurements

# MAIN =============================================================================================

def main() -> None:
    from gradus.registration import NETWORK_REGISTRY

    args = parse_arguments()
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output) / f"seed-{args.seed}_{stamp}"
    output.mkdir(parents = True, exist_ok = False)
    dense = NETWORK_REGISTRY.load_network(
        network_id = "resnet-18", input_shape = (3, 32, 32), num_classes = 100
    ).to(device).train()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location = device, weights_only = True)
        for key in ("state_dict", "model_state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        dense.load_state_dict(state, strict = True)
    sampled = deepcopy(dense)
    replace_convolutions(sampled)
    for layer in sampled.modules():
        if isinstance(layer, SampledConv2d):
            layer.mode = args.mode
    assert list(dict(dense.named_parameters())) == list(dict(sampled.named_parameters()))

    # Fixed data and parameters isolate estimator noise; this is not a learning experiment.
    x = torch.randn(args.batch_size, 3, 32, 32, device = device)
    y = torch.randint(100, (args.batch_size,), device = device)
    print(f"Frozen ResNet-18; mode {args.mode}; {device}; batch {args.batch_size}; output: {output}", flush = True)
    dense_logits = backward_step(dense, x, y)
    reference = gradient_vector(dense).clone()
    if not bool(torch.isfinite(reference).all()) or float(reference.norm()) <= 0:
        raise RuntimeError("Dense reference gradient is nonfinite or zero; cannot compute relative errors.")
    sampled_logits = backward_step(sampled, x, y)
    q1_error = float((gradient_vector(sampled) - reference).norm() / reference.norm().clamp_min(1e-30))
    logits_error = float((sampled_logits - dense_logits).abs().max())
    bn_error = max(float((b - dict(sampled.named_buffers())[name]).abs().max())
                   for name, b in dense.named_buffers())
    if not all(isfinite(value) for value in (q1_error, logits_error, bn_error)) \
       or q1_error > 1e-4 or logits_error > 1e-5 or bn_error > 1e-5:
        raise RuntimeError(f"Dense-equivalence check failed: gradient={q1_error}, logits={logits_error}, BN={bn_error}")
    print(f"Dense-equivalence check passed: relative gradient error {q1_error:.3e}", flush = True)

    rows: List[Dict] = []
    methods = {"native_dense": (dense, 1.0), "custom_dense": (sampled, 1.0)}
    for fraction in args.fractions:
        name = f"sampled_{fraction:g}"
        methods[name] = (sampled, fraction)
        set_fraction(sampled, fraction)
        print(f"Gradient diagnostic: {name}", flush = True)
        rows.append({"method": name, "requested_fraction": fraction,
                     "actual_fraction": ceil(fraction * args.batch_size) / args.batch_size,
                     **gradient_statistics(sampled, x, y, reference, args.trials)})
    del reference
    dense.zero_grad(set_to_none = True)
    sampled.zero_grad(set_to_none = True)
    print("Synchronized forward/backward wall-time profiling:", flush = True)
    timings = profile(methods, x, y, args)
    native_ms = median(timings["native_dense"])
    for name in ("native_dense", "custom_dense"):
        rows.insert(0 if name == "native_dense" else 1,
                    {"method": name, "requested_fraction": 1.0, "actual_fraction": 1.0})
    for row in rows:
        values = timings[row["method"]]
        row.update({"median_forward_backward_ms": median(values), "min_forward_backward_ms": min(values),
                    "max_forward_backward_ms": max(values), "speedup_vs_native": native_ms / median(values),
                    "wall_savings_pct": 100 * (1 - median(values) / native_ms)})

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(output / "results.csv", "w", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = fields)
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "arguments": vars(args), "torch": torch.__version__, "python": platform.python_version(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "custom_convolutions": sum(isinstance(m, SampledConv2d) for m in sampled.modules()),
        "q1_relative_gradient_error": q1_error, "q1_max_logits_error": logits_error,
        "q1_max_bn_buffer_error": bn_error, "raw_timing_ms": timings,
        "scope": "Frozen synthetic batch, train-mode BN, float32, no optimizer, data loading or learning evaluation.",
        "mode": args.mode,
    }
    with open(output / "metadata.json", "w") as handle:
        dump(metadata, handle, indent = 2)

    lines = ["SELECTIVE BACKWARD: DENSE-ADJOINT CONTROL", "", metadata["scope"],
             f"Sampling mode: {args.mode}; weight mode keeps all propagated input adjoints dense.",
             f"Device: {metadata['device_name']}; Torch: {torch.__version__}; batch: {args.batch_size}",
             f"Checkpoint: {args.checkpoint or 'random initialization'}; draws: {args.trials}",
             f"Dense-equivalence relative gradient error: {q1_error:.3e}", "",
             "method             keep    ms/step   speedup   saving%   mean_err   MCSE_norm  draw_RMS   cosine"]
    for row in rows:
        values = "  ".join(f"{row[key]:9.4g}" if key in row else f"{'-':>9}"
                           for key in ("relative_mean_error", "relative_mcse_norm", "relative_single_draw_rms", "mean_gradient_cosine"))
        lines.append(f"{row['method']:18} {row['actual_fraction']:.3f} {row['median_forward_backward_ms']:10.3f} "
                     f"{row['speedup_vs_native']:8.3f} {row['wall_savings_pct']:9.2f} {values}")
    lines += ["", "Speedups compare with native dense autograd, not the slower custom dense path.",
              "Mean error and MCSE norm are relative to the dense gradient norm; MCSE is not a confidence interval.",
              "Finite-trial mean error is not a pass/fail test of unbiasedness. Large draw_RMS signals noisy gradients.",
              "Convolution parameter gradients are sampled independently; BatchNorm backward remains native and complete.",
              "Weight mode computes dense convolution input adjoints; joint mode also samples input adjoints.",
              "BN running buffers evolve, but train-mode outputs use this same fixed batch's statistics.",
              "Synthetic random-state profiling cannot establish CIFAR accuracy, novelty or training speedup.",
              "Timings include zero_grad, full forward, loss, mask generation, gather/scatter and backward.",
              "They exclude gradient diagnostics, data loading and optimizer updates.", "", f"Output: {output}"]
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    print("\n" + summary, flush = True)


if __name__ == "__main__":
    main()
