"""# Stage-10 Confirmation Summary (CPU only)

Headline table for the method with an ESTIMATED noise rate, against full-data CE, full-data GCE and
random exclusion at the same budget the oracle rule would use. Only runs with --trusted-size > 0
are used, so every arm trains on exactly the same samples.

## Tables
    1. Headline: mean +/- std final accuracy per arm and condition, method deltas (seed-paired), compute.
    2. Estimator: noise rate read by the method (p_hat) vs the true rate, and the budget it chose.
    3. Sensitivity: the rule fed a deliberately wrong noise rate (--rule-noise), vs the estimate.

## Usage (from the gradus repo root)
    python scratch_confirm_summary.py

## Output
    analysis/pruning/confirm_summary.txt - share this one.
"""

from argparse   import ArgumentParser
from pathlib    import Path

import numpy    as np
import pandas   as pd

COND =  ["dataset", "network", "label_noise", "noise_type"]

def arm(row) -> str:
    """# Readable arm name for one results row."""
    if row.score == "none":         return f"full {row.objective.upper()}"
    if row.budget == "rule":
        if row.rule_noise < 0:      return "METHOD (estimated p)"
        return "rule, true p" if np.isclose(row.rule_noise, row.label_noise) else f"rule, p = {row.rule_noise:.2f}"
    if row.score == "random":       return "random"
    return f"{row.score} {row.drop} T={row.temperature:g} @ {row.prune_fraction:.2f}"

def main() -> None:
    p =         ArgumentParser(description = "Summarize the stage-10 confirmation runs.")
    p.add_argument("--root",    type = str, default = "analysis/pruning")
    args =      p.parse_args()
    root =      Path(args.root)

    frames = []
    for f in sorted(root.glob("*/*/results.csv")):
        df =    pd.read_csv(f, keep_default_na = False, na_values = [""], dtype = {"run_id": str})
        if "trusted_size" not in df: continue
        frames.append(df[df["trusted_size"].fillna(0) > 0])
    if not frames or not sum(len(f) for f in frames): print(f"No runs with --trusted-size under {root}"); return
    df =        pd.concat(frames, ignore_index = True)
    for c, d in (("budget", "fixed"), ("rule_noise", -1.0), ("objective", "ce"), ("noise_type", "symmetric")):
        df[c] = df[c].fillna(d)
    df["arm"] = [arm(r) for r in df.itertuples()]
    df["compute"] = 1 - df["dsi"]
    pd.set_option("display.width", 250)
    lines =     [f"{len(df)} runs with a trusted split (sizes: {sorted(df['trusted_size'].unique().tolist())}). "
                 "Accuracies and deltas in percentage points.", ""]

    # 1. Headline: per-arm accuracy and seed-paired method deltas.
    acc =       df.pivot_table(index = COND + ["seed"], columns = "arm", values = "final_acc", aggfunc = "mean")
    method =    "METHOD (estimated p)"
    rows = []
    for key, g in df.groupby(COND):
        r =     dict(zip(COND, key))
        for a, ga in g.groupby("arm"):
            if a.startswith("rule, p ="): continue
            r[a] = f"{100 * ga['final_acc'].mean():.2f} +/- {100 * ga['final_acc'].std(ddof = 1):.2f} (n={len(ga)})" \
                   if len(ga) > 1 else f"{100 * ga['final_acc'].mean():.2f} (n=1)"
        if method in acc.columns:
            sub =   acc.loc[key] if key in acc.index.droplevel("seed") else None
            if sub is not None:
                for ref in ("full CE", "full GCE", "random", "rule, true p"):
                    if ref in sub.columns:
                        d =     (sub[method] - sub[ref]).dropna()
                        r[f"method - {ref}"] = f"{100 * d.mean():+.2f} +/- {100 * d.std(ddof = 1) / np.sqrt(len(d)):.2f} (n={len(d)})" \
                                               if len(d) > 1 else (f"{100 * d.mean():+.2f} (n=1)" if len(d) else "")
            m =     g[g["arm"] == method]
            if len(m): r["method compute %"] = round(100 * m["compute"].mean(), 1)
        rows.append(r)
    head =      pd.DataFrame(rows)
    arms =      [a for a in ("full CE", "full GCE", "random", "rule, true p", method) if a in head.columns]
    deltas =    [c for c in head.columns if c.startswith("method")]
    lines +=    ["== 1. HEADLINE: final accuracy (mean +/- std over seeds) ==",
                 head[COND + arms].fillna("").to_string(index = False), "",
                 "== 1b. METHOD deltas (seed-paired mean +/- standard error) and compute (% of full-data samples) ==",
                 head[COND + deltas].fillna("").to_string(index = False), ""]

    # 2. Estimator accuracy as used by the method.
    m =         df[df["arm"] == method]
    if len(m):
        est =   m.assign(err = m["rule_p"] - m["label_noise"]).groupby(COND).agg(
                    n =             ("seed", "count"),
                    p_hat =         ("rule_p", "mean"),
                    p_hat_min =     ("rule_p", "min"),
                    p_hat_max =     ("rule_p", "max"),
                    abs_err =       ("err", lambda e: e.abs().mean()),
                    budget =        ("rule_rho", "mean"),
                    tilted =        ("rule_T", lambda t: (t > 0).mean()),
                    noisy_excl =    ("noisy_excl", "mean"),
                ).reset_index()
        for c in ("p_hat", "p_hat_min", "p_hat_max", "abs_err", "budget", "noisy_excl"): est[c] = (100 * est[c]).round(1)
        lines += ["== 2. ESTIMATOR as used by the method (x100; tilted = fraction of seeds where p_hat >= 0.1) ==",
                  est.to_string(index = False),
                  f"overall MAE = {100 * (m['rule_p'] - m['label_noise']).abs().mean():.1f} points over {len(m)} runs", ""]

    # 3. Sensitivity of the rule to a wrong noise rate.
    s =         df[(df["budget"] == "rule")]
    if (s["rule_noise"] >= 0).any():
        s =     s.assign(fed = np.where(s["rule_noise"] < 0, s["rule_p"], s["rule_noise"]),
                         source = np.where(s["rule_noise"] < 0, "estimated", "fixed"))
        t =     s.groupby(COND + ["source", "rule_noise"]).agg(
                    n =         ("seed", "count"),
                    p_fed =     ("fed", "mean"),
                    budget =    ("rule_rho", "mean"),
                    final_acc = ("final_acc", "mean"),
                    std =       ("final_acc", "std"),
                ).reset_index().sort_values(COND + ["p_fed"])
        t["p_error"] = t["p_fed"] - t["label_noise"]
        for c in ("p_fed", "p_error", "budget", "final_acc", "std"): t[c] = (100 * t[c]).round(2)
        lines += ["== 3. SENSITIVITY: rule fed the noise rate p_fed (p_error = p_fed - true noise; x100) ==",
                  t.drop(columns = "rule_noise").to_string(index = False), ""]

    text =      "\n".join(lines)
    out =       root / "confirm_summary.txt"
    out.write_text(text);   print(text);    print(f"\nWritten to {out.absolute()}")

if __name__ == "__main__": main()
