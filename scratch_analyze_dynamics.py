"""# Training-Dynamics Follow-Up Analysis (CPU only; reuses diagnose_metrics.py outputs)

Answers four design questions for a dynamic, metric-aware sample pruner:

    A. TIMING     Do static metrics predict per-sample loss at ANY point in training (e.g. early),
                  even though they do not predict it on average?
    B. ONSET      How many epochs of dynamics are needed before EL2N / AUM / loss give a reliable,
                  predictive difficulty ranking (i.e. when can a dynamic pruner safely act)?
    C. STABILITY  Is a per-sample activation-stability signal (sample-level B-PAS) informative once
                  normalized for activation magnitude, and in which phase of training?
    D. COVERAGE   How much does pruning by a dynamic score distort class balance & the distribution
                  of image statistics - and does stratifying by class x metric-bin remove that?

## Usage (from the gradus repo root, after diagnose_metrics.py has cached >= 1 seed)
    python analyze_dynamics.py --dataset cifar-100 --network resnet-18 --epochs 100

## Outputs (written next to the cached dynamics)
    followup_summary.txt    All tables - share this one.
    followup_*.csv          Individual tables.
    followup_curves.png     Per-epoch metric correlations (A) and early-score reliability (B).
"""

from argparse   import ArgumentParser, Namespace
from pathlib    import Path
from typing     import Dict, List, Tuple

import numpy    as np
import pandas   as pd

# ARGUMENTS ========================================================================================

def parse_arguments() -> Namespace:
    """# Parse Analysis Arguments."""
    parser = ArgumentParser(description = "Follow-up analysis of cached per-sample training dynamics.")
    parser.add_argument("--network",        type = str,     default = "resnet-18")
    parser.add_argument("--dataset",        type = str,     default = "cifar-100")
    parser.add_argument("--epochs",         type = int,     default = 100)
    parser.add_argument("--output",         type = str,     default = "analysis/diagnostics")
    parser.add_argument("--scores",         type = str,     default = None,
                        help = "Static scores parquet. Defaults to .cache/scores/<dataset>/metric-scores_seed-1.parquet")
    parser.add_argument("--strata-metric",  type = str,     default = "spatial-frequency",
                        help = "Static metric whose quantile bins (x class) define pruning strata in part D.")
    parser.add_argument("--strata-bins",    type = int,     default = 5)
    parser.add_argument("--onset-k",        type = int,     default = 10,
                        help = "Epochs of EL2N used as the pruning score in part D.")
    return parser.parse_args()

# CORRELATION HELPERS ==============================================================================

def _zscore_ranks_(matrix: np.ndarray) -> np.ndarray:
    """# Column-wise average ranks, standardized (constant columns -> NaN)."""
    ranks = pd.DataFrame(matrix).rank().to_numpy()
    std =   ranks.std(axis = 0)
    std[std == 0] = np.nan
    return (ranks - ranks.mean(axis = 0)) / std

def within_class_spearman(X: np.ndarray, Y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """# Mean over classes of the Spearman matrix between columns of X (N x p) and Y (N x q)."""
    X, Y =      X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    blocks =    []
    for c in np.unique(labels):
        idx =   labels == c
        if idx.sum() < 10: continue
        blocks.append(_zscore_ranks_(X[idx]).T @ _zscore_ranks_(Y[idx]) / idx.sum())
    return np.nanmean(np.stack(blocks), axis = 0)

def within_class_partial(a: np.ndarray, b: np.ndarray, s: np.ndarray, labels: np.ndarray) -> float:
    """# Mean over classes of the partial Spearman of a & b controlling for s."""
    values = []
    for c in np.unique(labels):
        idx =   labels == c
        if idx.sum() < 10: continue
        r =     np.corrcoef(_zscore_ranks_(np.column_stack([a[idx], b[idx], s[idx]])), rowvar = False)
        r_ab, r_as, r_bs = r[0, 1], r[0, 2], r[1, 2]
        values.append((r_ab - r_as * r_bs) / np.sqrt((1 - r_as ** 2) * (1 - r_bs ** 2)))
    return float(np.nanmean(values))

def diag_within(X: np.ndarray, Y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """# Within-class Spearman between matching columns of X & Y (e.g. per-epoch cross-seed)."""
    return np.array([within_class_spearman(X[:, j], Y[:, j], labels)[0, 0] for j in range(X.shape[1])])

# A. TIMING ========================================================================================

def analyze_timing(traj: Dict[int, dict], static: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """# Per-epoch within-class Spearman of each static metric with that epoch's (seed-mean) loss."""
    seeds =     sorted(traj)
    loss =      np.mean([traj[s]["loss"] for s in seeds], axis = 0)                # N x E
    table =     pd.DataFrame(
                    within_class_spearman(static.to_numpy(), loss, labels).T,       # E x metrics
                    columns =   static.columns,
                    index =     pd.Index(np.arange(1, loss.shape[1] + 1), name = "epoch")
                )

    # Noise ceiling: max attainable |rho| with the seed-mean loss of that epoch.
    if len(seeds) >= 2:
        r =                     diag_within(traj[seeds[0]]["loss"], traj[seeds[1]]["loss"], labels)
        mean_reliability =      np.clip(2 * r / (1 + r), 0, 1)                      # Spearman-Brown
        table["ceiling"] =      np.sqrt(mean_reliability)
    return table

# B. ONSET =========================================================================================

def analyze_onset(traj: Dict[int, dict], labels: np.ndarray) -> pd.DataFrame:
    """# Reliability & predictive validity of difficulty scores built from the first k epochs.

    * reliability   Within-class Spearman of the k-epoch score between two seeds.
    * validity      Within-class Spearman of the k-epoch score (one seed) with the FULL-run mean loss
                    of an independent seed - i.e. how well an early score predicts eventual difficulty
                    without sharing noise. (Falls back to the same seed when only one is cached.)
    """
    seeds =     sorted(traj)
    E =         traj[seeds[0]]["loss"].shape[1]
    ks =        [k for k in (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100) if k <= E]
    other =     {s: seeds[(i + 1) % len(seeds)] for i, s in enumerate(seeds)}
    rows =      []

    for k in ks:
        row = {"k": k}
        for name, key, sign in (("el2n", "el2n", 1), ("aum", "margin", -1), ("loss", "loss", 1)):
            early = {s: sign * np.nanmean(traj[s][key][:, :k], axis = 1) for s in seeds}
            final = {s: np.nanmean(traj[s]["loss"], axis = 1) for s in seeds}
            row[f"{name}_validity"] = float(np.mean([
                within_class_spearman(early[s], final[other[s]], labels)[0, 0] for s in seeds
            ]))
            if len(seeds) >= 2:
                row[f"{name}_reliability"] = float(within_class_spearman(early[seeds[0]], early[seeds[1]], labels)[0, 0])
        rows.append(row)

    return pd.DataFrame(rows).set_index("k")

# C. STABILITY =====================================================================================

def stability_variants(act: np.ndarray, window: slice) -> Dict[str, np.ndarray]:
    """# Per-sample activation-stability signals over an epoch window."""
    a =     act[:, window]
    d =     np.abs(np.diff(a, axis = 1))
    return  {
                "abs_delta":    np.nanmean(d, axis = 1),
                "rel_delta":    np.nanmean(d / np.maximum(a[:, :-1], 1e-8), axis = 1),
                "cv":           np.nanstd(a, axis = 1) / np.maximum(np.nanmean(a, axis = 1), 1e-8),
            }

def analyze_stability(traj: Dict[int, dict], labels: np.ndarray) -> pd.DataFrame:
    """# Reliability & difficulty-relevance of normalized activation-stability signals by phase."""
    seeds =     sorted(traj)
    E =         traj[seeds[0]]["act_std"].shape[1]
    windows =   {"early (1-10)": slice(0, min(10, E)), "mid": slice(E // 3, 2 * E // 3), "late": slice(2 * E // 3, E), "all": slice(0, E)}
    other =     {s: seeds[(i + 1) % len(seeds)] for i, s in enumerate(seeds)}
    rows =      []

    for wname, window in windows.items():
        if window.stop - window.start < 3: continue
        variants =  {s: stability_variants(traj[s]["act_std"], window) for s in seeds}
        magnitude = {s: np.nanmean(traj[s]["act_std"][:, window], axis = 1) for s in seeds}
        final =     {s: np.nanmean(traj[s]["loss"], axis = 1) for s in seeds}
        for vname in variants[seeds[0]]:
            row = {"window": wname, "signal": vname}
            row["rho_with_final_loss"] =    float(np.mean([
                within_class_spearman(variants[s][vname], final[other[s]], labels)[0, 0] for s in seeds
            ]))
            row["partial_rho_given_magnitude"] = float(np.mean([
                within_class_partial(variants[s][vname], final[other[s]], magnitude[s], labels) for s in seeds
            ]))
            if len(seeds) >= 2:
                row["reliability"] = float(within_class_spearman(variants[seeds[0]][vname], variants[seeds[1]][vname], labels)[0, 0])
            rows.append(row)

    return pd.DataFrame(rows).set_index(["window", "signal"])

# D. COVERAGE ======================================================================================

def ks_statistic(sample: np.ndarray, reference: np.ndarray) -> float:
    """# Two-sample Kolmogorov-Smirnov statistic (sup |ECDF difference|)."""
    sample, reference = np.sort(sample[~np.isnan(sample)]), np.sort(reference[~np.isnan(reference)])
    grid =  np.concatenate([sample, reference])
    return  float(np.max(np.abs(
                np.searchsorted(sample, grid, side = "right") / len(sample) -
                np.searchsorted(reference, grid, side = "right") / len(reference)
            )))

def prune_mask(score: np.ndarray, fraction: float, policy: str, groups: np.ndarray, rng) -> np.ndarray:
    """# Boolean KEEP mask removing `fraction` of samples, optionally within each group."""
    keep =  np.ones(len(score), dtype = bool)
    for g in (np.unique(groups) if groups is not None else [None]):
        idx =   np.flatnonzero(groups == g) if g is not None else np.arange(len(score))
        n =     int(round(fraction * len(idx)))
        if n == 0: continue
        if   policy == "random":        drop = rng.choice(idx, size = n, replace = False)
        elif policy == "drop_easy":     drop = idx[np.argsort(score[idx])[:n]]
        elif policy == "drop_hard":     drop = idx[np.argsort(score[idx])[::-1][:n]]
        keep[drop] = False
    return keep

def analyze_coverage(traj: Dict[int, dict], static: pd.DataFrame, labels: np.ndarray, args: Namespace) -> pd.DataFrame:
    """# Distortion of class balance & static-metric distributions under dynamic-score pruning."""
    rng =       np.random.default_rng(0)
    seeds =     sorted(traj)
    score =     np.mean([np.nanmean(traj[s]["el2n"][:, :args.onset_k], axis = 1) for s in seeds], axis = 0)

    # Strata: class x quantile bin of the strata metric.
    metric =    static[args.strata_metric].to_numpy()
    bins =      pd.Series(metric).groupby(labels).transform(
                    lambda v: pd.qcut(v.rank(method = "first"), args.strata_bins, labels = False)
                ).to_numpy()
    groupings = {"none": None, "class": labels, f"class x {args.strata_metric} bin": labels * args.strata_bins + bins}

    classes =   np.unique(labels)
    rows =      []
    for fraction in (0.3, 0.5, 0.7):
        for policy in ("random", "drop_easy", "drop_hard"):
            for gname, groups in groupings.items():
                if policy == "random" and gname != "none": continue
                keep =  prune_mask(score, fraction, policy, groups, rng)
                kept_classes = np.array([(labels[keep] == c).mean() for c in classes])
                ks =    {m: ks_statistic(static[m].to_numpy()[keep], static[m].to_numpy()) for m in static.columns}
                rows.append({
                    "pruned":           fraction,
                    "policy":           policy,
                    "stratified_by":    gname,
                    "class_tv":         0.5 * np.abs(kept_classes - 1 / len(classes)).sum(),
                    "min_class_kept":   min((labels[keep] == c).sum() / (labels == c).sum() for c in classes),
                    "mean_ks":          np.mean(list(ks.values())),
                    "max_ks":           max(ks.values()),
                    "max_ks_metric":    max(ks, key = ks.get),
                    "kept_mean_final_loss": float(np.mean([np.nanmean(traj[s]["loss"], axis = 1)[keep].mean() for s in seeds])),
                })
    return pd.DataFrame(rows).set_index(["pruned", "policy", "stratified_by"])

# REPORTING ========================================================================================

def plot_curves(timing: pd.DataFrame, onset: pd.DataFrame, path: Path) -> None:
    """# A: per-epoch metric correlations vs ceiling.  B: early-score reliability/validity vs k."""
    from matplotlib             import use;     use("Agg")
    from matplotlib.pyplot      import close, subplots

    palette =   ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, (ax_a, ax_b) = subplots(1, 2, figsize = (13, 4.6))

    metrics =   [c for c in timing.columns if c != "ceiling"]
    for color, m in zip(palette, metrics):
        ax_a.plot(timing.index, timing[m], color = color, linewidth = 2, label = m)
    if "ceiling" in timing:
        ax_a.plot(timing.index, timing["ceiling"], color = "#8a8985", linewidth = 1.5, linestyle = "--", label = "noise ceiling")
        ax_a.plot(timing.index, -timing["ceiling"], color = "#8a8985", linewidth = 1.5, linestyle = "--")
    ax_a.axhline(0, color = "#c9c8c4", linewidth = 1)
    ax_a.set(title = "A. Within-class ρ(static metric, loss at epoch)", xlabel = "epoch", ylabel = "Spearman ρ")
    ax_a.legend(fontsize = 7, frameon = False, ncol = 2)

    for color, name in zip(palette, ("el2n", "aum", "loss")):
        ax_b.plot(onset.index, onset[f"{name}_validity"], color = color, linewidth = 2, marker = "o", markersize = 4, label = f"{name} validity")
        if f"{name}_reliability" in onset:
            ax_b.plot(onset.index, onset[f"{name}_reliability"], color = color, linewidth = 2, linestyle = ":", label = f"{name} reliability")
    ax_b.set_xscale("log")
    ax_b.set(title = "B. Score from first k epochs", xlabel = "k (epochs, log scale)", ylabel = "within-class Spearman ρ")
    ax_b.legend(fontsize = 7, frameon = False, ncol = 2)

    for ax in (ax_a, ax_b):
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.grid(axis = "y", color = "#ecebe8", linewidth = 0.8)

    fig.tight_layout();     fig.savefig(path, dpi = 150);   close(fig)

def main() -> None:
    """# Load Cached Dynamics, Run A-D, & Report."""
    args =          parse_arguments()
    directory =     Path(args.output) / args.dataset / f"{args.network}_epochs-{args.epochs}"
    scores_path =   Path(args.scores or f".cache/scores/{args.dataset}/metric-scores_seed-1.parquet")

    traj =          {int(f.stem.rsplit("-", 1)[1]): dict(np.load(f)) for f in sorted(directory.glob("dynamics_seed-*.npz"))}
    if not traj:    raise RuntimeError(f"No cached dynamics in {directory}")
    first =         traj[min(traj)]
    labels =        first["labels"]

    static =        pd.read_parquet(scores_path).set_index("index").dropna(axis = 1, how = "all")
    static =        static.reindex(first["indices"]).drop(columns = "class")
    print(f"Seeds {sorted(traj)}; {len(labels)} samples; metrics {list(static.columns)}")

    pd.set_option("display.width", 250)
    tables: Dict[str, pd.DataFrame] = {}
    print("A. timing...");      tables["timing"] =      analyze_timing(traj, static, labels)
    print("B. onset...");       tables["onset"] =       analyze_onset(traj, labels)
    print("C. stability...");   tables["stability"] =   analyze_stability(traj, labels)
    print("D. coverage...");    tables["coverage"] =    analyze_coverage(traj, static, labels, args)

    for name, table in tables.items(): table.to_csv(directory / f"followup_{name}.csv")
    plot_curves(tables["timing"], tables["onset"], directory / "followup_curves.png")

    timing =        tables["timing"]
    metric_cols =   [c for c in timing.columns if c != "ceiling"]
    headline =      pd.DataFrame({
                        "max_|rho|_any_epoch":  timing[metric_cols].abs().max(),
                        "at_epoch":             timing[metric_cols].abs().idxmax(),
                        "mean_|rho|_epochs_1-5": timing[metric_cols].iloc[:5].abs().mean(),
                        "mean_|rho|_all":       timing[metric_cols].abs().mean(),
                    })
    sampled =       timing.loc[[e for e in (1, 2, 3, 5, 10, 20, 30, 50, 75, 100) if e in timing.index]]

    lines = [
        f"network = {args.network}; dataset = {args.dataset}; epochs = {args.epochs}; seeds = {sorted(traj)}",
        f"strata = class x {args.strata_bins} bins of {args.strata_metric}; pruning score = EL2N over first {args.onset_k} epochs",
        "",
        "== A. TIMING: headline per static metric (within-class rho with per-epoch loss) ==", headline.round(3).to_string(), "",
        "== A. TIMING: sampled epochs ==", sampled.round(3).to_string(), "",
        "== B. ONSET: early-score reliability & validity (aum sign-flipped so + = hard) ==", tables["onset"].round(3).to_string(), "",
        "== C. STABILITY: normalized activation-stability signals ==", tables["stability"].round(3).to_string(), "",
        "== D. COVERAGE: distortion from pruning by EL2N ==", tables["coverage"].round(3).to_string(), "",
    ]
    text = "\n".join(lines)
    (directory / "followup_summary.txt").write_text(text)
    print(text)
    print(f"\nResults written to {directory.absolute()}")

if __name__ == "__main__": main()