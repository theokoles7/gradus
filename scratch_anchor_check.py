"""# Anchored Noise-Rate Estimator Check (CPU only)

Tests the estimator suggested by the calibration structure  m = p + (1 - p) * e_clean :

    p_hat = (m - e) / (1 - e)

where m is the training negative-margin fraction and e is the error on a held-out set with TRUSTED
(clean) labels at the same epoch. The calibration runs already logged the error on the clean CIFAR
test split every epoch, so this emulates a small trusted held-out set offline. (In the final method
e would come from a carved-out trusted split - never from the test set.)

## Usage (from the gradus repo root)
    python scratch_anchor_check.py

## Output
    analysis/pruning/anchor_check.txt - share this one.
"""

from argparse   import ArgumentParser
from json       import load
from pathlib    import Path

import numpy    as np
import pandas   as pd

EPOCHS = (5, 8, 10, 12, 15)

def main() -> None:
    p =         ArgumentParser(description = "Check the trusted-set-anchored noise-rate estimator.")
    p.add_argument("--root", type = str, default = "analysis/pruning")
    args =      p.parse_args()
    root =      Path(args.root)

    rows = []
    for f in sorted(root.glob("*/*/calibration/*_stop*.json")):
        rec =   load(open(f))
        for h in rec["history"]:
            if h["epoch"] not in EPOCHS: continue
            m, e =  h["est_negmargin"], 1 - h["val_acc"]
            rows.append({"dataset": rec["dataset"], "network": rec["network"],
                         "type": rec.get("noise_type", "symmetric"), "noise": rec.get("label_noise", 0.0),
                         "seed": rec["seed"], "epoch": h["epoch"], "negmargin": m, "heldout_err": e,
                         "raw": m, "anchored": (m - e) / (1 - e) if e < 1 else np.nan})
    if not rows: print(f"No calibration records under {root}"); return

    df =        pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    lines =     [f"{df.groupby(['dataset', 'network', 'type', 'noise', 'seed']).ngroups} calibration runs.", ""]

    board = []
    for e, g in df.groupby("epoch"):
        clean = g[g["noise"] == 0]
        for name in ("raw", "anchored"):
            err =   g[name] - g["noise"]
            board.append({"epoch": e, "estimator": name, "MAE_identity_x100": 100 * err.abs().mean(),
                          "max_abs_err_x100": 100 * err.abs().max(), "bias_x100": 100 * err.mean(),
                          "bias_clean_x100": 100 * clean[name].mean() if len(clean) else np.nan,
                          "sym_MAE_x100": 100 * err[g["type"] == "symmetric"].abs().mean(),
                          "pairflip_MAE_x100": 100 * err[g["type"] == "pairflip"].abs().mean()})
    lines += ["== SCOREBOARD (identity: no calibration) ==",
              pd.DataFrame(board).round(2).to_string(index = False), ""]

    for e, g in df.groupby("epoch"):
        t = g.groupby(["dataset", "network", "type", "noise"])[["negmargin", "heldout_err", "anchored"]].mean().mul(100).round(1)
        lines += [f"== epoch {e}: training negmargin, held-out (trusted) error, anchored estimate (x100) ==",
                  t.to_string(), ""]

    text =      "\n".join(lines)
    out =       root / "anchor_check.txt"
    out.write_text(text);   print(text);    print(f"\nWritten to {out.absolute()}")

if __name__ == "__main__": main()
