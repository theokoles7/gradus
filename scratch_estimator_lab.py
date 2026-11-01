"""# Noise-Rate Estimator Lab (CPU only)

Evaluates candidate label-noise-rate estimators offline on the per-sample snapshots written by
`scratch_prune_ablation.py --stop-after K --snapshot-epochs ...` (calibration/<run_id>_ep<E>.npz).

The question is portability: does an estimator read the true noise rate directly (identity), with
no per-dataset calibration, across datasets, architectures and noise types?

## Candidates
    negmargin       fraction of samples with negative EMA margin
    gmm_loss        high-mean weight of a 2-Gaussian mixture on EMA loss
    gmm_margin      low-mean weight of a 2-Gaussian mixture on EMA margin
    gmm_margin_cls  gmm_margin fitted within each (given) class, averaged over classes
    bmm_loss        high-mean weight of a 2-Beta mixture on normalized EMA loss (Arazo et al., 2019)
    disagree@t      fraction whose last prediction != given label with confidence > t (t = 0.5, 0.7, 0.9)

## Metrics (per snapshot epoch)
    MAE_identity    mean |estimate - true noise| with NO calibration (the portable case)
    R2_pooled, MAE_pooled   after one linear fit pooled over all datasets/architectures
    bias_clean      estimate on clean data (should be ~0)

## Usage (from the gradus repo root)
    python scratch_estimator_lab.py

## Output
    analysis/pruning/estimator_lab.txt - share this one.
"""

from argparse   import ArgumentParser
from json       import load
from math       import lgamma
from pathlib    import Path

import numpy    as np
import pandas   as pd

# MIXTURES =========================================================================================

def gmm_1d(x: np.ndarray, iterations: int = 60):
    """# 2-component 1-D Gaussian mixture (plain EM). Returns (weights, means)."""
    x =         x[np.isfinite(x)].astype(np.float64)
    lo, hi =    np.quantile(x, [0.25, 0.75])
    mu =        np.array([lo, hi]);     var = np.full(2, x.var() + 1e-6);     w = np.array([0.5, 0.5])
    for _ in range(iterations):
        lp =    -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2 * np.pi * var)) + np.log(w)
        lp -=   lp.max(axis = 1, keepdims = True)
        r =     np.exp(lp);     r /= r.sum(axis = 1, keepdims = True)
        nk =    r.sum(axis = 0) + 1e-12
        w =     nk / len(x);    mu = (r * x[:, None]).sum(axis = 0) / nk
        var =   (r * (x[:, None] - mu) ** 2).sum(axis = 0) / nk + 1e-6
    return w, mu

def _beta_logpdf_(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - (lgamma(a) + lgamma(b) - lgamma(a + b))

def bmm_1d(x: np.ndarray, iterations: int = 30):
    """# 2-component Beta mixture on loss normalized to (0, 1) (Arazo et al., 2019). Returns (weights, means)."""
    x =         x[np.isfinite(x)].astype(np.float64)
    lo, hi =    np.percentile(x, [5, 95])
    x =         np.clip((x - lo) / max(hi - lo, 1e-12), 1e-4, 1 - 1e-4)
    r =         np.stack([x < 0.5, x >= 0.5], axis = 1).astype(np.float64) + 1e-3
    r /=        r.sum(axis = 1, keepdims = True)
    for _ in range(iterations):
        nk =    r.sum(axis = 0) + 1e-12;    w = nk / len(x)
        m =     (r * x[:, None]).sum(axis = 0) / nk
        v =     (r * (x[:, None] - m) ** 2).sum(axis = 0) / nk + 1e-8
        common = np.clip(m * (1 - m) / v - 1, 1e-2, None)
        a, b =  np.clip(m * common, 1e-2, 1e3), np.clip((1 - m) * common, 1e-2, 1e3)
        lp =    np.stack([_beta_logpdf_(x, a[k], b[k]) for k in range(2)], axis = 1) + np.log(w)
        lp -=   lp.max(axis = 1, keepdims = True)
        r =     np.exp(lp);     r /= r.sum(axis = 1, keepdims = True)
    return w, a / (a + b)

# ESTIMATORS =======================================================================================

def estimators(snap) -> dict:
    """# All candidate noise-rate estimates for one snapshot."""
    loss, margin, given = snap["ema_loss"], snap["ema_margin"], snap["given"]
    pred, conf =    snap["pred"], snap["conf"]
    out =           {"negmargin": float(np.nanmean(margin < 0))}

    w, mu =         gmm_1d(loss);       out["gmm_loss"] =   float(w[np.argmax(mu)])
    w, mu =         gmm_1d(margin);     out["gmm_margin"] = float(w[np.argmin(mu)])
    per_class =     []
    for c in np.unique(given):
        m =         margin[given == c]
        if np.isfinite(m).sum() >= 50:
            w, mu = gmm_1d(m);  per_class.append(w[np.argmin(mu)])
    out["gmm_margin_cls"] = float(np.mean(per_class)) if per_class else np.nan
    w, mu =         bmm_1d(loss);       out["bmm_loss"] =   float(w[np.argmax(mu)])

    seen =          pred >= 0
    for t in (0.5, 0.7, 0.9):
        out[f"disagree@{t}"] = float(((pred != given) & (conf > t) & seen).sum() / max(1, seen.sum()))
    return out

# MAIN =============================================================================================

def fit(x, y):
    if len(x) < 3 or np.ptp(x) == 0: return np.nan, np.nan
    b, a =  np.polyfit(x, y, 1);    pred = a + b * x
    ss =    ((y - y.mean()) ** 2).sum()
    return (1 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan), np.abs(y - pred).mean()

def main() -> None:
    p =         ArgumentParser(description = "Offline noise-rate estimator evaluation.")
    p.add_argument("--root", type = str, default = "analysis/pruning")
    args =      p.parse_args()
    root =      Path(args.root)

    rows = []
    for f in sorted(root.glob("*/*/calibration/*_ep*.npz")):
        run_id, ep =    f.stem.rsplit("_ep", 1)
        meta =          next(iter(sorted(f.parent.glob(f"{run_id}_stop*.json"))), None)
        info =          load(open(meta)) if meta else {}
        snap =          dict(np.load(f))
        rows.append({
            "dataset":  f.parent.parent.parent.name, "network": f.parent.parent.name,
            "type":     info.get("noise_type", "symmetric"), "noise": float(snap["noisy"].mean()),
            "nominal":  info.get("label_noise", np.nan), "seed": info.get("seed", np.nan), "epoch": int(ep),
            **estimators(snap),
        })
    if not rows: print(f"No snapshots under {root}"); return

    df =        pd.DataFrame(rows)
    names =     [c for c in df.columns if c not in ("dataset", "network", "type", "noise", "nominal", "seed", "epoch")]
    pd.set_option("display.width", 250)
    lines =     [f"{len(df)} snapshots from {df.groupby(['dataset', 'network', 'type', 'nominal', 'seed']).ngroups} runs.", ""]

    # Scoreboard: portability of each estimator at each epoch.
    board = []
    for e, g in df.groupby("epoch"):
        clean = g[g["noise"] == 0]
        for n in names:
            r2, mae = fit(g[n].to_numpy(), g["noise"].to_numpy())
            board.append({"epoch": e, "estimator": n,
                          "MAE_identity_x100": 100 * np.abs(g[n] - g["noise"]).mean(),
                          "R2_pooled": r2, "MAE_pooled_x100": 100 * mae,
                          "bias_clean_x100": 100 * clean[n].mean() if len(clean) else np.nan})
    lines += ["== SCOREBOARD (lower MAE_identity = more portable; R2/MAE_pooled after one pooled linear fit) ==",
              pd.DataFrame(board).sort_values(["epoch", "MAE_identity_x100"]).round(3).to_string(index = False), ""]

    # Per-condition estimates.
    for e, g in df.groupby("epoch"):
        t = g.groupby(["dataset", "network", "type", "nominal"])[names].mean().mul(100).round(1)
        lines += [f"== estimates (x100) at epoch {e}; compare with the nominal noise rate ==", t.to_string(), ""]

    text =      "\n".join(lines)
    out =       root / "estimator_lab.txt"
    out.write_text(text);   print(text);    print(f"\nWritten to {out.absolute()}")

if __name__ == "__main__": main()