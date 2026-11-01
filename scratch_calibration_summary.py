"""# Noise-Estimator Calibration Summary

Reads every calibration record written by `scratch_prune_ablation.py --stop-after K` (under
analysis/pruning/<dataset>/<network>/calibration/) and reports, per estimator:

    1. Seed-averaged estimates (x100) per condition at each early epoch.
    2. Per epoch, a least-squares fit  true_noise ~ a + b * estimate  within each dataset/network
       and pooled across all of them. A portable estimator has similar (a, b) everywhere and a high
       pooled R^2; MAE is the mean absolute error of the pooled fit, in noise-rate points.
    3. AUC of each EMA score for identifying label-flipped samples (noisy conditions only).

## Usage (from the gradus repo root)
    python scratch_calibration_summary.py

## Output
    analysis/pruning/calibration_summary.txt - share this one.
"""

from argparse   import ArgumentParser
from json       import load
from pathlib    import Path

import numpy    as np
import pandas   as pd

ESTIMATORS =    {"negmargin": "est_negmargin", "el2n_gt1": "est_el2n_gt1", "gmm": "est_gmm"}
AUCS =          {"loss": "auc_loss", "el2n": "auc_el2n", "negmargin": "auc_negmargin"}

def fit(x: np.ndarray, y: np.ndarray):
    """# Least squares y = a + b x; returns (a, b, R^2, MAE)."""
    if len(x) < 3 or np.ptp(x) == 0: return (np.nan,) * 4
    b, a =      np.polyfit(x, y, 1)
    pred =      a + b * x
    ss_res =    ((y - pred) ** 2).sum();    ss_tot = ((y - y.mean()) ** 2).sum()
    return a, b, 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, np.abs(y - pred).mean()

def main() -> None:
    """# Collect, Fit, & Report."""
    p =         ArgumentParser(description = "Summarize noise-estimator calibration runs.")
    p.add_argument("--root", type = str, default = "analysis/pruning")
    args =      p.parse_args()
    root =      Path(args.root)

    rows = []
    for f in sorted(root.glob("*/*/calibration/*.json")):
        rec =   load(open(f))
        for h in rec["history"]:
            rows.append({
                "dataset":  rec["dataset"], "network": rec["network"], "seed": rec["seed"],
                "noise":    rec.get("label_noise", 0.0), "type": rec.get("noise_type", "symmetric"),
                "epoch":    h["epoch"],
                **{k: h.get(v, np.nan) for k, v in ESTIMATORS.items()},
                **{f"auc_{k}": h.get(v, np.nan) for k, v in AUCS.items()},
            })
    if not rows: print(f"No calibration records under {root}"); return

    df =        pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    epochs =    [e for e in (3, 5, 8, 10, 12, 15, 20) if e in set(df["epoch"])]
    df =        df[df["epoch"].isin(epochs)]
    cond =      ["dataset", "network", "type", "noise"]
    lines =     [f"{df.groupby(cond + ['seed']).ngroups} calibration runs; epochs {epochs[0]}-{epochs[-1]}.", ""]

    # 1. Estimates per condition and epoch.
    for est in ESTIMATORS:
        t = df.pivot_table(index = cond, columns = "epoch", values = est, aggfunc = "mean").mul(100).round(1)
        lines += [f"== {est}: seed-averaged estimate (x100) by epoch ==", t.to_string(), ""]

    # 2. Linear calibration against the true noise rate, per group and pooled.
    for est in ESTIMATORS:
        fits = []
        for e in epochs:
            sub =   df[df["epoch"] == e].groupby(cond)[est].mean().reset_index()
            for (ds, net), g in list(sub.groupby(["dataset", "network"])) + [(("POOLED", ""), sub)]:
                a, b, r2, mae = fit(g[est].to_numpy(), g["noise"].to_numpy())
                fits.append({"epoch": e, "dataset": ds, "network": net, "a": a, "b": b, "R2": r2,
                             "MAE_x100": mae * 100})
        t = pd.DataFrame(fits).round(3)
        lines += [f"== {est}: fit true_noise ~ a + b * estimate ==", t.to_string(index = False), ""]

    # 3. AUC for identifying flipped samples.
    noisy =     df[df["noise"] > 0]
    if len(noisy):
        for k in AUCS:
            t = noisy.pivot_table(index = cond, columns = "epoch", values = f"auc_{k}", aggfunc = "mean").round(3)
            lines += [f"== AUC of EMA {k} for flagging flipped labels (0.5 = chance) ==", t.to_string(), ""]

    text =      "\n".join(lines)
    out =       root / "calibration_summary.txt"
    out.write_text(text);   print(text);    print(f"\nWritten to {out.absolute()}")

if __name__ == "__main__": main()
