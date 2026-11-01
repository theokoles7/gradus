"""# scratch_cgspc_diagnostic.py

Go/no-go diagnostic for the CGSPC proposal (persistent per-example
gradient-consensus agreement as a curriculum signal), BEFORE building
the actual controller. Pure measurement — training is always plain,
unweighted, fully shuffled (identical to scratch_reweight.py's
--unweighted mode), under label noise. Nothing here alters a single
gradient update; it only records, for every sample every epoch, five
candidate difficulty/utility signals, then reports how well each one
separates clean from corrupted examples at a handful of checkpoints.

Five signals, per sample i:
  1. instantaneous loss                 l_i           (this epoch's value)
  2. instantaneous gradient norm        ||G_i||_F = ||delta_i|| * ||h_i||
  3. instantaneous directional agreement with the current per-class
     gradient consensus:                a_i = cos(G_i, M_{y_i})
  4. temporal (EMA) agreement           s_i = beta_s*s_i + (1-beta_s)*a_i
  5. temporally-smoothed loss           l_smoothed_i (EMA of l_i)
     — the cheap control condition. If this alone separates as well as
     the full directional-agreement machinery, most of CGSPC's value is
     coming from persistence, not direction, and a much simpler
     mechanism (one EMA line added to scratch_reweight.py) would do.

Per-example classifier-layer gradient:
    G_i = delta_i (x) h_i     (outer product, rank 1)
    delta_i = softmax(logits_i) - onehot(y_i),   shape (num_classes,)
    h_i = penultimate feature, shape (feat_dim,)
h_i is captured via a forward hook on the network's LAST nn.Linear
layer, so this works without needing to know this specific ResNet-18
implementation's internal attribute names.

Per-TRUE-class consensus M_c (shape (num_classes, feat_dim), same shape
as any single G_i) is an EMA of the UNWEIGHTED per-batch average G_i for
samples whose true label is c, updated from the plain shuffled stream.
Agreement for a batch is computed against M as it stood BEFORE that
batch's update is folded in (per the CGSPC design, to avoid
self-reinforcing feedback), then M is updated afterward.

cos(G_i, M_{y_i}) never requires materializing G_i's full outer product
for the *comparison* (only for the *update* of M, once per batch): since
G_i is rank-1,
    <G_i, M_c>_F = delta_i^T (M_c h_i)      (a bilinear form)
    ||G_i||_F    = ||delta_i|| * ||h_i||    (rank-1 norm factorizes)
Only ||M_c||_F needs an explicit Frobenius norm, computed once per class
per update.

Separation is reported as AUC (clean vs. corrupted, Mann-Whitney
rank-sum formula, no external dependency) at each checkpoint epoch, plus
max(auc, 1-auc) as a sign-agnostic separation score in [0.5, 1], since
loss/gradient-norm correlate positively with "corrupted" while agreement
signals correlate negatively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing   import Any, Dict, List, Optional

import numpy as np
import torch
from torch                  import device as t_device, no_grad, Tensor
from torch.nn.functional    import cross_entropy
from torch.optim            import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data       import DataLoader, Dataset as TorchDataset
from tqdm                   import tqdm

from gradus.datasets         import Dataset
from gradus.networks         import Network
from gradus.registration     import DATASET_REGISTRY, NETWORK_REGISTRY
from gradus.utilities        import determine_device, get_logger, set_seed


class IndexedNoisyLabelDataset(TorchDataset):
    """# Wraps a Dataset with Label Noise, Also Returning the Sample Index.

    Same corruption logic and seed-decoupling rationale as
    scratch_reweight.py's NoisyLabelDataset (noise_seed is fixed,
    separate from the training seed, so every run compares against the
    identical corruption pattern). Also returns the dataset index
    alongside (image, label), needed to persist per-sample state across
    epochs despite the DataLoader reshuffling batch order every epoch.
    """

    def __init__(self, base_dataset: Any, num_classes: int, noise_rate: float, noise_seed: int):
        self._base_ = base_dataset
        n = len(base_dataset)
        rng = np.random.RandomState(noise_seed)
        noisy_mask = rng.rand(n) < noise_rate

        labels = np.array([base_dataset[i][1] for i in range(n)])
        for i in np.where(noisy_mask)[0]:
            true_label = labels[i]
            wrong = rng.randint(0, num_classes)
            while wrong == true_label:
                wrong = rng.randint(0, num_classes)
            labels[i] = wrong

        self._labels_ = labels
        self.is_corrupted = noisy_mask
        self.actual_noise_rate = float(noisy_mask.mean())

    def __len__(self) -> int:
        return len(self._base_)

    def __getitem__(self, idx: int):
        image, _ = self._base_[idx]
        return image, int(self._labels_[idx]), idx


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """# AUC via the Mann-Whitney Rank-Sum Formula. No External Dependency.

    labels: 1 = corrupted (positive class), 0 = clean.
    """
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype = np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run(
    network_id:            str,
    dataset_id:             str,
    batch_size:             int =              128,
    epochs:                 int =              100,
    seed:                   int =              1,
    label_noise_rate:       float =            0.3,
    label_noise_seed:       int =              1,
    consensus_beta:         float =            0.9,
    temporal_beta:          float =            0.9,
    smoothed_loss_beta:     float =            0.9,
    checkpoint_epochs:      Optional[List[int]] = None,
    device:                 str =              "auto",
    output_path:            str =              "results/cgspc_diagnostic",
) -> Dict[str, Any]:
    """# Run the CGSPC Go/No-Go Diagnostic. No Training Intervention."""
    logger = get_logger("cgspc-diagnostic")

    if checkpoint_epochs is None:
        checkpoint_epochs = [5, 10, 20, 30, 50, 70, 100]
    checkpoint_set = set(e for e in checkpoint_epochs if e <= epochs)

    set_seed(seed = seed)
    resolved_device: t_device = determine_device(device = device)
    print(f"Device: {resolved_device}")

    dataset: Dataset = DATASET_REGISTRY.load_dataset(
        dataset_id = dataset_id,
        seed =        seed,
        epochs =      epochs,
        batch_size =  batch_size,
        metric =      None
    )

    network: Network = NETWORK_REGISTRY.load_network(
        network_id =  network_id,
        input_shape = dataset.input_shape,
        num_classes = dataset.num_classes
    ).to(resolved_device)

    num_classes = dataset.num_classes

    train_data = IndexedNoisyLabelDataset(
        base_dataset = dataset.train_data, num_classes = num_classes,
        noise_rate = label_noise_rate, noise_seed = label_noise_seed
    )
    print(f"Label noise: requested {label_noise_rate:.2%}, actual {train_data.actual_noise_rate:.2%} "
                f"(noise_seed={label_noise_seed}, independent of training seed={seed}).")

    train_loader = DataLoader(
        dataset =      train_data,
        batch_size =   batch_size,
        shuffle =      True,
        num_workers =  dataset._max_workers_,
        pin_memory =   True
    )
    test_loader = dataset.test_loader

    optimizer = SGD(params = network.parameters(), lr = 0.1, weight_decay = 5e-4, momentum = 0.9)
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    # Forward hook on the LAST nn.Linear layer, capturing its input
    # (the penultimate feature h_i).
    classifier_layer = None
    for module in network.modules():
        if isinstance(module, torch.nn.Linear):
            classifier_layer = module
    if classifier_layer is None:
        raise RuntimeError("No nn.Linear layer found — cannot capture penultimate features.")

    captured: Dict[str, Tensor] = {}

    def _capture_hook(module, inp, outp):
        captured["h"] = inp[0].detach()

    hook_handle = classifier_layer.register_forward_hook(_capture_hook)

    n = len(train_data)
    is_corrupted = train_data.is_corrupted.astype(np.int64)

    # Persistent per-sample state across epochs.
    temporal_agreement = np.zeros(n, dtype = np.float64)
    smoothed_loss       = np.zeros(n, dtype = np.float64)
    state_initialized    = np.zeros(n, dtype = bool)   # first observation seeds state directly

    # Transient per-epoch instantaneous values — overwritten exactly once
    # per sample per epoch (standard shuffling gives full, non-overlapping
    # coverage), so at epoch end these hold that epoch's instantaneous value.
    inst_loss           = np.zeros(n, dtype = np.float64)
    inst_grad_norm       = np.zeros(n, dtype = np.float64)
    inst_agreement       = np.zeros(n, dtype = np.float64)

    feat_dim: Optional[int] = None
    M: Optional[Tensor] = None                      # (num_classes, num_classes, feat_dim)
    M_seen = np.zeros(num_classes, dtype = bool)
    eps = 1e-8

    epoch_records: List[Dict[str, Any]] = []
    checkpoint_aucs: Dict[int, Dict[str, Any]] = {}

    for epoch in range(1, epochs + 1):
        network.train()
        train_total_loss, train_correct, train_total = 0.0, 0, 0

        for samples, targets, indices in tqdm(train_loader, desc = f"Epoch {epoch}/{epochs}", leave = False):
            samples, targets = samples.to(resolved_device), targets.to(resolved_device)
            idx_np = indices.numpy()

            predictions: Tensor = network(samples)
            loss = cross_entropy(input = predictions, target = targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with no_grad():
                h = captured["h"]                                    # (B, feat_dim)
                if feat_dim is None:
                    feat_dim = h.shape[1]
                    M = torch.zeros(num_classes, num_classes, feat_dim, device = resolved_device)

                probs = predictions.softmax(dim = 1)
                one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
                delta = probs - one_hot                               # (B, num_classes)
                per_sample_loss = cross_entropy(input = predictions, target = targets, reduction = "none")

                delta_norm = delta.norm(dim = 1)                      # (B,)
                h_norm = h.norm(dim = 1)                              # (B,)
                grad_norm = delta_norm * h_norm                       # ||G_i||_F, rank-1 factorization

                # Agreement against the PRE-update consensus.
                M_frob = M.reshape(num_classes, -1).norm(dim = 1)     # (num_classes,) — one Frobenius norm per class
                M_y = M[targets]                                      # (B, num_classes, feat_dim)
                Mh = torch.einsum("bcf,bf->bc", M_y, h)                # (B, num_classes) = M_{y_i} @ h_i per sample
                numerator = torch.einsum("bc,bc->b", delta, Mh)        # (B,) = delta_i^T (M_{y_i} h_i)
                denom = delta_norm * h_norm * M_frob[targets] + eps
                agreement = numerator / denom
                # Classes never yet seen have M_c == 0 (M_frob == 0), giving
                # an uninformative but harmless agreement ~= 0 via the eps
                # guard, rather than a NaN from 0/0.

                # Update M from this batch's average G_i per true class,
                # AFTER computing this batch's agreement above.
                G = torch.einsum("bc,bf->bcf", delta, h)               # (B, num_classes, feat_dim)
                unique_classes = targets.unique()
                for c in unique_classes.tolist():
                    mask = targets == c
                    g_c_mean = G[mask].mean(dim = 0)                   # (num_classes, feat_dim)
                    if not M_seen[c]:
                        M[c] = g_c_mean
                        M_seen[c] = True
                    else:
                        M[c] = consensus_beta * M[c] + (1 - consensus_beta) * g_c_mean

                # Persist per-sample state.
                loss_np = per_sample_loss.cpu().numpy()
                agreement_np = agreement.cpu().numpy()
                grad_norm_np = grad_norm.cpu().numpy()

                inst_loss[idx_np] = loss_np
                inst_grad_norm[idx_np] = grad_norm_np
                inst_agreement[idx_np] = agreement_np

                first_time = ~state_initialized[idx_np]
                idx_first = idx_np[first_time]
                idx_rest = idx_np[~first_time]

                if len(idx_first) > 0:
                    temporal_agreement[idx_first] = agreement_np[first_time]
                    smoothed_loss[idx_first] = loss_np[first_time]
                    state_initialized[idx_first] = True
                if len(idx_rest) > 0:
                    temporal_agreement[idx_rest] = (
                        temporal_beta * temporal_agreement[idx_rest]
                        + (1 - temporal_beta) * agreement_np[~first_time]
                    )
                    smoothed_loss[idx_rest] = (
                        smoothed_loss_beta * smoothed_loss[idx_rest]
                        + (1 - smoothed_loss_beta) * loss_np[~first_time]
                    )

            train_total_loss += loss.item()
            _, predicted = predictions.max(dim = 1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        network.eval()
        val_correct, val_total = 0, 0
        with no_grad():
            for samples, targets in test_loader:
                samples, targets = samples.to(resolved_device), targets.to(resolved_device)
                predictions = network(samples)
                _, predicted = predictions.max(dim = 1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        val_accuracy = round(val_correct / val_total, 4)

        record: Dict[str, Any] = {
            "epoch":          epoch,
            "train_accuracy": round(train_correct / train_total, 4),
            "val_accuracy":   val_accuracy,
        }
        epoch_records.append(record)

        if epoch in checkpoint_set:
            signals = {
                "inst_loss":          inst_loss,
                "inst_grad_norm":     inst_grad_norm,
                "inst_agreement":     inst_agreement,
                "temporal_agreement": temporal_agreement,
                "smoothed_loss":      smoothed_loss,
            }
            aucs: Dict[str, Any] = {}
            for name, sig in signals.items():
                a = auc_score(sig, is_corrupted)
                aucs[name] = {"auc": round(a, 4), "separation": round(max(a, 1 - a), 4)}
            checkpoint_aucs[epoch] = aucs
            print(f"Epoch {epoch:3} checkpoint — separation (0.5=none, 1.0=perfect): " +
                        ", ".join(f"{k}={v['separation']:.3f}" for k, v in aucs.items()))

    hook_handle.remove()

    result: Dict[str, Any] = {
        "config": {
            "network_id":         network_id,
            "dataset_id":         dataset_id,
            "batch_size":         batch_size,
            "epochs":             epochs,
            "seed":               seed,
            "label_noise_rate":   label_noise_rate,
            "label_noise_seed":   label_noise_seed,
            "consensus_beta":     consensus_beta,
            "temporal_beta":      temporal_beta,
            "smoothed_loss_beta": smoothed_loss_beta,
            "checkpoint_epochs":  sorted(checkpoint_set),
        },
        "epochs":          epoch_records,
        "checkpoint_aucs": checkpoint_aucs,
    }

    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)
    tag = f"seed-{seed}_cgspc-diag-noise{label_noise_rate}"
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent = 2)
    print(f"Saved: {out_dir / f'{tag}.json'}")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "CGSPC go/no-go signal-separation diagnostic.")
    parser.add_argument("--network-id",   type = str,   default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-100")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 100)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument("--label-noise-rate", type = float, default = 0.3)
    parser.add_argument("--label-noise-seed", type = int,   default = 1)
    parser.add_argument("--consensus-beta",     type = float, default = 0.9)
    parser.add_argument("--temporal-beta",      type = float, default = 0.9)
    parser.add_argument("--smoothed-loss-beta", type = float, default = 0.9)
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/cgspc_diagnostic")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        network_id =          args.network_id,
        dataset_id =           args.dataset_id,
        batch_size =           args.batch_size,
        epochs =               args.epochs,
        seed =                 args.seed,
        label_noise_rate =     args.label_noise_rate,
        label_noise_seed =     args.label_noise_seed,
        consensus_beta =       args.consensus_beta,
        temporal_beta =        args.temporal_beta,
        smoothed_loss_beta =   args.smoothed_loss_beta,
        device =               args.device,
        output_path =          args.output_path,
    )