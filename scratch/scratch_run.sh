#!/usr/bin/env bash
# Two-day queue. Claim-critical runs first; resumable (existing outputs skipped).

set -u
cd "$(dirname "$0")"
mkdir -p results/bpas results/bpas_stale logs

# stale pre-fix baseline (77.14) contaminates frontier averages
mv results/bpas/baseline_seed*.json results/bpas_stale/ 2>/dev/null || true

run () {   # run <arm> <dsi|none> <seed> <outfile> [extra args...]
  local arm="$1" dsi="$2" seed="$3" out="results/bpas/$4"; shift 4
  if [ -f "$out" ]; then echo "skip  $(basename "$out")"; return; fi
  echo "=== $(date +%H:%M:%S)  $(basename "$out")"
  local dsiarg=(); [ "$dsi" != "none" ] && dsiarg=(--dsi-target "$dsi")
  python scratch_bpas.py --seed "$seed" --arm "$arm" "${dsiarg[@]}" "$@" \
      --output-path "$out" >> "logs/$(basename "$out").log" 2>&1
  [ -f "$out" ] || echo "FAILED $(basename "$out")  see logs/"
}

sat () { echo --static-scores ".cache/scores/metric-scores_seed-$1.parquet" \
              --static-metric saturation-time --prune-side low; }

# 1. harden the headline -----------------------------------------------------
for s in 4 5 6 7 8 9; do
  run baseline      none "$s" "v2_baseline_seed$s.json"
  run sample_random 0.25 "$s" "sample_random_dsi0.25_seed$s.json"
  run sample_random 0.40 "$s" "sample_random_dsi0.40_seed$s.json"
done

# 2. head-to-head with B-PAS at its own DSI ----------------------------------
for s in 1 2 3 4 5 6 7 8 9; do
  run sample_random 0.0662 "$s" "sample_random_dsi0.0662_seed$s.json"
done

# 3. curriculum criterion in the regime where it could help ------------------
#    sample-level, reshuffled composition, high DSI
for s in 1 2 3; do
  for d in 0.40 0.55; do
    run sample_complexity "$d" "$s" "sample_cplx_sat_dsi${d}_seed$s.json" $(sat "$s")
    run sample_accuracy   "$d" "$s" "sample_acc_dsi${d}_seed$s.json"
    run sample_random     "$d" "$s" "sample_random_dsi${d}_seed$s.json"
  done
done

# 4. extend the frontier -----------------------------------------------------
for s in 1 2 3; do
  for d in 0.10 0.70; do
    run sample_random "$d" "$s" "sample_random_dsi${d}_seed$s.json"
  done
done

# results in one file --------------------------------------------------------
{
  echo "generated $(date)"
  echo "runs on disk: $(ls results/bpas/*.json 2>/dev/null | wc -l)"
  echo
  python scratch_bpas.py --analyze "results/bpas/*.json"
  echo
  echo "=== per-file summary ==="
  python - <<'PY'
import glob, json
rows=[]
for f in sorted(glob.glob("results/bpas/*.json")):
    d=json.load(open(f))
    rows.append((d["arm"], d.get("static_metric") or "", 100*d["dsi"],
                 d["seed"], d["best_test_acc"], d["epochs_run"]))
print(f"{'arm':<19}{'metric':<16}{'DSI%':>7}{'seed':>6}{'best':>8}{'epochs':>8}")
for r in sorted(rows):
    print(f"{r[0]:<19}{r[1]:<16}{r[2]:>7.2f}{r[3]:>6}{r[4]:>8.2f}{r[5]:>8}")
PY
} > results/SUMMARY.txt 2>&1

echo; echo "done. paste results/SUMMARY.txt into the chat."; cat results/SUMMARY.txt