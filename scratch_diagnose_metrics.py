"""# Metric Diagnostic Experiment

Do static difficulty metrics carry information about how samples actually behave in training?

1. RECORD: Train <network> on <dataset> under the standard i.i.d. protocol (reshuffled every
   epoch; same SGD + cosine schedule as `gradus train`) and log, for every training sample at
   every epoch, its loss, correctness, margin, EL2N, & mean activation std. One .npz per seed.
2. ANALYZE: Derive per-sample dynamic scores and rank-correlate them with every static metric in
   the dataset's score parquet - globally, within class, & between classes - alongside the
   cross-seed reliability (noise ceiling) of each dynamic score.

## Dynamic scores
    * el2n          Mean ||softmax - onehot||_2 over first k epochs (Paul et al., 2021). High = hard.
    * loss_auc      Mean per-sample loss over all epochs. High = hard.
    * aum           Mean margin (true logit - max other logit) (Pleiss et al., 2020). High = EASY.
    * forgetting    Correct -> incorrect transitions (Toneva et al., 2019); never-learned = max+1.
    * learned_epoch First epoch after which the sample stays correct; E+1 if never. High = hard.
    * act_std       Mean per-sample activation std over tracked layers & epochs.
    * act_delta     Mean |epoch-to-epoch change| in act_std (sample-level B-PAS signal).

## Usage (from the gradus repo root)
    python diagnose_metrics.py --network resnet-18 --dataset cifar-100 --seeds 1 2
    python diagnose_metrics.py --network resnet-18 --dataset cifar-100 --seeds 1 --subset 2000 --epochs 3   # smoke test
    python diagnose_metrics.py --network resnet-18 --dataset cifar-100 --seeds 1 2 --analyze-only           # re-analyze cached runs

## Outputs (under --output/<dataset>/<network>_epochs-<E>/)
    dynamics_seed-<s>.npz   Raw per-sample, per-epoch trajectories (reusable for offline pruning sims).
    <table>.csv             global, within_class, between_class, dynamic_dynamic, static_static, reliability.
    correlations.png        Heatmaps of the three static-vs-dynamic decompositions.
    summary.txt             Everything above in one plain-text file - share this one.
"""

from argparse   import ArgumentParser, Namespace
from itertools  import combinations
from pathlib    import Path
from typing     import Dict, List, Optional

import numpy    as np
import pandas   as pd

DYNAMIC_SCORES: List[str] = [
                                "el2n", "loss_auc", "aum", "forgetting", "learned_epoch",
                                "act_std", "act_delta"
                            ]

# ARGUMENTS ========================================================================================

def parse_arguments() -> Namespace:
    """# Parse Experiment Arguments."""
    parser = ArgumentParser(description = "Correlate static difficulty metrics with training dynamics.")
    parser.add_argument("--network",        type = str,     default = "resnet-18")
    parser.add_argument("--dataset",        type = str,     default = "cifar-100")
    parser.add_argument("--seeds",          type = int,     default = [1, 2], nargs = "+",
                        help = "Training seeds to record (>= 2 enables the reliability table).")
    parser.add_argument("--epochs",         type = int,     default = 100)
    parser.add_argument("--el2n-epochs",    type = int,     default = 10)
    parser.add_argument("--batch-size",     type = int,     default = 128)
    parser.add_argument("--workers",        type = int,     default = 4)
    parser.add_argument("--root",           type = str,     default = ".cache/data")
    parser.add_argument("--scores",         type = str,     default = None,
                        help = "Static scores parquet. Defaults to .cache/scores/<dataset>/metric-scores_seed-1.parquet")
    parser.add_argument("--output",         type = str,     default = "analysis/diagnostics")
    parser.add_argument("--device",         type = str,     default = "auto", choices = ["auto", "cpu", "cuda"])
    parser.add_argument("--subset",         type = int,     default = None,
                        help = "Only use the first N training samples (smoke testing).")
    parser.add_argument("--analyze-only",   action = "store_true",
                        help = "Skip training; analyze whatever seeds are already cached.")
    return parser.parse_args()

# RECORDING ========================================================================================

def record_dynamics(args: Namespace, seed: int, path: Path) -> None:
    """# Train Under the i.i.d. Protocol & Record Per-Sample Trajectories."""
    from torch                      import Generator, no_grad, stack
    from torch.nn.functional        import cross_entropy, one_hot, softmax
    from torch.optim                import SGD
    from torch.optim.lr_scheduler   import CosineAnnealingLR
    from torch.utils.data           import DataLoader, Dataset, Subset
    from tqdm                       import tqdm

    from gradus.registration        import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities           import determine_device, set_seed

    class Indexed(Dataset):
        """Yield (sample, target, index)."""
        def __init__(self, data):       self.data = data
        def __len__(self):              return len(self.data)
        def __getitem__(self, i):       x, y = self.data[i]; return x, y, i

    set_seed(seed);     device = determine_device(args.device)

    # Gradus dataset with no curriculum; we only borrow its splits & test loader.
    dataset =           DATASET_REGISTRY.load_dataset(
                            dataset_id =    args.dataset,
                            epochs =        args.epochs,
                            seed =          seed,
                            root =          args.root,
                            batch_size =    args.batch_size,
                            max_workers =   args.workers
                        )
    train_data =        dataset.train_data if args.subset is None \
                        else Subset(dataset.train_data, range(min(args.subset, len(dataset.train_data))))
    N, E =              len(train_data), args.epochs
    loader =            DataLoader(
                            Indexed(train_data),
                            batch_size =    args.batch_size,
                            shuffle =       True,
                            num_workers =   args.workers,
                            pin_memory =    True,
                            generator =     Generator().manual_seed(seed)
                        )

    network =           NETWORK_REGISTRY.load_network(
                            network_id =    args.network,
                            input_shape =   dataset.input_shape,
                            num_classes =   dataset.num_classes
                        ).to(device)
    optimizer =         SGD(network.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    scheduler =         CosineAnnealingLR(optimizer, T_max = E)

    rec =               {k: np.full((N, E), np.nan, dtype = np.float32)
                         for k in ("loss", "margin", "el2n", "act_std")}
    rec["correct"] =    np.zeros((N, E), dtype = bool)
    labels =            np.full(N, -1, dtype = np.int64)
    val_accuracy =      []

    print(f"[seed {seed}] Recording {args.network} on {args.dataset}: {N} samples x {E} epochs on {device}")

    for epoch in range(E):

        network.train()
        for x, y, idx in tqdm(loader, desc = f"[seed {seed}] epoch {epoch + 1}/{E}", leave = False):

            x, y =                  x.to(device), y.to(device)
            logits, activations =   network(x, return_activations = True)
            per_sample =            cross_entropy(logits, y, reduction = "none")
            optimizer.zero_grad();  per_sample.mean().backward();   optimizer.step()

            with no_grad():
                z =                 logits.detach()
                onehot =            one_hot(y, num_classes = z.shape[1]).float()
                true_logit =        z.gather(1, y.unsqueeze(1)).squeeze(1)
                other_max =         z.masked_fill(onehot.bool(), float("-inf")).max(dim = 1).values
                stds =              stack([a.flatten(1).std(dim = 1) for a in activations]).mean(dim = 0)

                rows =                          idx.numpy()
                rec["loss"][rows, epoch] =      per_sample.detach().cpu().numpy()
                rec["el2n"][rows, epoch] =      (softmax(z, dim = 1) - onehot).norm(dim = 1).cpu().numpy()
                rec["margin"][rows, epoch] =    (true_logit - other_max).cpu().numpy()
                rec["correct"][rows, epoch] =   z.argmax(dim = 1).eq(y).cpu().numpy()
                rec["act_std"][rows, epoch] =   stds.cpu().numpy()
                labels[rows] =                  y.cpu().numpy()

        scheduler.step()

        # Validation, to confirm the recorded run is a normal run.
        network.eval(); hits = total = 0
        with no_grad():
            for x, y in dataset.test_loader:
                hits +=     network(x.to(device)).argmax(dim = 1).eq(y.to(device)).sum().item()
                total +=    y.size(0)
        val_accuracy.append(hits / total)
        print(f"[seed {seed}] epoch {epoch + 1:3}/{E}: train acc = {rec['correct'][:, epoch].mean():.4f}; "
              f"val acc = {val_accuracy[-1]:.4f}")

    path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(path, **rec, labels = labels, indices = np.arange(N),
                        val_accuracy = np.array(val_accuracy, dtype = np.float32))
    print(f"[seed {seed}] Saved dynamics to {path}")

# DERIVATION =======================================================================================

def derive_scores(trajectory: Dict[str, np.ndarray], el2n_epochs: int) -> pd.DataFrame:
    """# Reduce Per-Epoch Trajectories to Per-Sample Dynamic Scores."""
    correct =       trajectory["correct"]
    E =             correct.shape[1]

    forgetting =    (correct[:, :-1] & ~correct[:, 1:]).sum(axis = 1).astype(float)
    never =         ~correct.any(axis = 1)
    if never.any(): forgetting[never] = forgetting.max() + 1

    incorrect =     ~correct
    last_miss =     E - 1 - incorrect[:, ::-1].argmax(axis = 1)
    learned_epoch = np.where(incorrect.any(axis = 1), last_miss + 2, 1).astype(float)

    return  pd.DataFrame(
                {
                    "el2n":             np.nanmean(trajectory["el2n"][:, :max(1, el2n_epochs)], axis = 1),
                    "loss_auc":         np.nanmean(trajectory["loss"], axis = 1),
                    "aum":              np.nanmean(trajectory["margin"], axis = 1),
                    "forgetting":       forgetting,
                    "learned_epoch":    learned_epoch,
                    "act_std":          np.nanmean(trajectory["act_std"], axis = 1),
                    "act_delta":        np.nanmean(np.abs(np.diff(trajectory["act_std"], axis = 1)), axis = 1),
                },
                index = pd.Index(trajectory["indices"], name = "index")
            )

# CORRELATION ======================================================================================

def spearman(frame: pd.DataFrame, rows: List[str], cols: List[str]) -> pd.DataFrame:
    """# Spearman (Pearson on Average Ranks) between Two Column Groups."""
    return frame[list(dict.fromkeys(rows + cols))].rank().corr().loc[rows, cols]

def correlate(static: pd.DataFrame, dynamics: Dict[int, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """# Decompose Static-vs-Dynamic Association (Global / Within-Class / Between-Class)."""
    mean_dynamics = pd.concat(dynamics.values()).groupby(level = 0).mean()
    metrics =       [c for c in static.columns if c != "class"]
    joined =        static.join(mean_dynamics, how = "inner")

    per_class =     [spearman(g, metrics, DYNAMIC_SCORES) for _, g in joined.groupby("class") if len(g) >= 10]

    tables = {
        "global":           spearman(joined, metrics, DYNAMIC_SCORES),
        "within_class":     pd.concat(per_class).groupby(level = 0).mean().loc[metrics, DYNAMIC_SCORES],
        "between_class":    spearman(joined.groupby("class")[metrics + DYNAMIC_SCORES].mean(), metrics, DYNAMIC_SCORES),
        "dynamic_dynamic":  spearman(joined, DYNAMIC_SCORES, DYNAMIC_SCORES),
        "static_static":    spearman(joined, metrics, metrics),
    }

    if len(dynamics) >= 2:
        tables["reliability"] = pd.DataFrame(
            {
                f"seed-{a}_vs_seed-{b}": [
                    dynamics[a][[c]].join(dynamics[b][[c]], rsuffix = "_b").rank().corr().iloc[0, 1]
                    for c in DYNAMIC_SCORES
                ]
                for a, b in combinations(sorted(dynamics), 2)
            },
            index = DYNAMIC_SCORES
        )

    return tables

# REPORTING ========================================================================================

def plot_correlations(tables: Dict[str, pd.DataFrame], path: Path, title: str) -> None:
    """# Heatmaps of the Three Static-vs-Dynamic Decompositions."""
    from matplotlib             import use;     use("Agg")
    from matplotlib.colors      import LinearSegmentedColormap
    from matplotlib.pyplot      import close, subplots

    # Diverging blue <-> red with a neutral gray midpoint.
    cmap =      LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
    panels =    ["global", "within_class", "between_class"]
    n_rows =    len(tables["global"])
    fig, axes = subplots(1, 3, figsize = (13, 0.45 * n_rows + 2.4), sharey = True)

    for ax, name in zip(axes, panels):
        table = tables[name]
        image = ax.imshow(table.values, cmap = cmap, vmin = -1, vmax = 1, aspect = "auto")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f"{table.values[i, j]:.2f}", ha = "center", va = "center", fontsize = 7, color = "#1a1a19")
        ax.set_title(name.replace("_", " "), fontsize = 10)
        ax.set_xticks(range(table.shape[1]), table.columns, rotation = 45, ha = "right", fontsize = 8)
        ax.set_yticks(range(table.shape[0]), table.index, fontsize = 8)
        for spine in ax.spines.values(): spine.set_visible(False)

    fig.colorbar(image, ax = axes, shrink = 0.8, label = "Spearman ρ")
    fig.suptitle(f"Static metrics vs. training dynamics: {title}", fontsize = 11)
    fig.savefig(path, dpi = 150, bbox_inches = "tight");    close(fig)

def write_summary(
    tables:     Dict[str, pd.DataFrame],
    dynamics:   Dict[int, pd.DataFrame],
    val_acc:    Dict[int, float],
    static:     pd.DataFrame,
    args:       Namespace,
    path:       Path
) -> str:
    """# Plain-Text Summary of Every Table plus Headline Numbers."""
    within =    tables["within_class"]
    headline =  pd.DataFrame({
                    "max_|rho|_within":     within.abs().max(axis = 1),
                    "argmax_dynamic":       within.abs().idxmax(axis = 1),
                    "max_|rho|_global":     tables["global"].abs().max(axis = 1),
                    "max_|rho|_between":    tables["between_class"].abs().max(axis = 1),
                })

    lines = [
        f"network = {args.network}; dataset = {args.dataset}; epochs = {args.epochs}; "
        f"el2n_epochs = {args.el2n_epochs}; subset = {args.subset}",
        f"seeds = {sorted(dynamics)}; final val acc = { {s: round(a, 4) for s, a in val_acc.items()} }",
        f"samples correlated = {len(static.join(next(iter(dynamics.values())), how = 'inner'))}; "
        f"classes = {static['class'].nunique()}; static metrics = {[c for c in static.columns if c != 'class']}",
        "",
        "== HEADLINE: strongest association of each static metric with any dynamic score ==",
        headline.round(3).to_string(),
        "",
    ]
    for name, table in tables.items():
        lines += [f"== {name.upper()} (Spearman rho) ==", table.round(3).to_string(), ""]

    text = "\n".join(lines)
    path.write_text(text)
    return text

# MAIN =============================================================================================

def main() -> None:
    """# Record (if needed) & Analyze."""
    args =          parse_arguments()
    tag =           f"{args.network}_epochs-{args.epochs}" + (f"_subset-{args.subset}" if args.subset else "")
    output_dir =    Path(args.output) / args.dataset / tag
    scores_path =   Path(args.scores or f".cache/scores/{args.dataset}/metric-scores_seed-1.parquet")

    # Fail fast if the static scores are missing (before spending GPU hours).
    if not scores_path.exists(): raise FileNotFoundError(f"No static scores at {scores_path}; pass --scores.")

    # 1. Record dynamics for every requested seed not already cached.
    for seed in args.seeds:
        npz = output_dir / f"dynamics_seed-{seed}.npz"
        if npz.exists():            print(f"[seed {seed}] Using cached dynamics: {npz}")
        elif args.analyze_only:     print(f"[seed {seed}] Not cached; skipping (--analyze-only)")
        else:                       record_dynamics(args, seed, npz)

    # 2. Load cached seeds & derive dynamic scores.
    trajectories =  {int(f.stem.rsplit("-", 1)[1]): dict(np.load(f))
                     for f in sorted(output_dir.glob("dynamics_seed-*.npz"))
                     if int(f.stem.rsplit("-", 1)[1]) in args.seeds}
    if not trajectories: raise RuntimeError(f"No cached dynamics in {output_dir}")
    dynamics =      {s: derive_scores(t, args.el2n_epochs) for s, t in trajectories.items()}
    val_acc =       {s: float(t["val_accuracy"][-1]) for s, t in trajectories.items()}

    # 3. Load static scores (drop metrics never computed).
    static =        pd.read_parquet(scores_path).set_index("index").dropna(axis = 1, how = "all")

    # 4. Correlate & report.
    tables =        correlate(static, dynamics)
    for name, table in tables.items(): table.to_csv(output_dir / f"{name}.csv")
    plot_correlations(tables, output_dir / "correlations.png", f"{args.dataset} / {tag}")
    print(write_summary(tables, dynamics, val_acc, static, args, output_dir / "summary.txt"))
    print(f"\nResults written to {output_dir.absolute()}")

if __name__ == "__main__": main()