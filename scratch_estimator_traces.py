"""# Noise-Estimator Traces

Reads every run JSON written by scratch_prune_ablation.py that logged online noise-rate estimates
(stage 7 onward) and tabulates, per condition, the seed-averaged estimates at selected epochs next
to the true noise rate. Used to calibrate the adaptive-budget rule.

## Usage (from the gradus repo root)
    python scratch_estimator_traces.py
    python scratch_estimator_traces.py --runs analysis/pruning/cifar-100/resnet-18/runs

## Output
    estimator_traces.txt (next to the runs directory) - share this one.
"""

from argparse   import ArgumentParser
from json       import load
from pathlib    import Path

import numpy    as np
import pandas   as pd

EPOCHS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def main() -> None:
    """# Collect, Aggregate, & Print Estimator Traces."""
    p =         ArgumentParser(description = "Tabulate online noise-estimate traces.")
    p.add_argument("--runs", type = str, default = "analysis/pruning/cifar-100/resnet-18/runs")
    args =      p.parse_args()
    runs_dir =  Path(args.runs)

    rows = []
    for f in sorted(runs_dir.glob("*.json")):
        run =   load(open(f))
        hist =  run.get("history", [])
        if not hist or "est_negmargin" not in hist[0]: continue
        cond =  {
                    "noise":    f"{run.get('label_noise', 0.0):.1f} {run.get('noise_type', 'symmetric')}",
                    "obj":      run.get("objective", "ce"),
                    "policy":   "full" if run["score"] == "none" else
                                f"{run['score']} T={run.get('temperature', 0)} rho={run['prune_fraction']}",
                }
        for h in hist:
            if h["epoch"] in EPOCHS:
                rows.append({**cond, "seed": run["seed"], "epoch": h["epoch"],
                             "negmargin": h["est_negmargin"], "gmm": h["est_gmm"],
                             "excluded": h["excluded_frac"]})

    if not rows: print(f"No runs with estimator logs under {runs_dir}"); return
    df =        pd.DataFrame(rows)
    pd.set_option("display.width", 250)

    blocks = [
        "Seed-averaged online noise-rate estimates (x100) at selected epochs.",
        "negmargin: % of samples with negative EMA margin.  gmm: high-loss component weight of a 2-component GMM.",
        "Compare against the true noise rate in each block header.", ""
    ]
    for (noise, obj, policy), g in df.groupby(["noise", "obj", "policy"]):
        t =     g.groupby("epoch")[["negmargin", "gmm", "excluded"]].mean().mul(100).round(1).T
        blocks += [f"== true noise {noise} | objective {obj} | {policy} | seeds {sorted(int(s) for s in g.seed.unique())} ==",
                   t.to_string(), ""]

    text = "\n".join(blocks)
    out =  runs_dir.parent / "estimator_traces.txt"
    out.write_text(text);   print(text);    print(f"\nWritten to {out.absolute()}")

if __name__ == "__main__": main()