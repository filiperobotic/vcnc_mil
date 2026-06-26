#!/usr/bin/env bash
# Orchestrator for the OA-MIL × VCNC-gate ablation sweep.
# 22 scenarios × 2 seeds = 44 runs total, ~1.5h each, ~66h wall clock.
#
# Resumable: if killed mid-sweep, rerun the script. It skips any
# (scenario, seed) pair already marked DONE in ablation_progress.txt.
#
# Invoke from the repo root, ideally inside a tmux session, WITH the
# conda env active:
#   conda activate mmlab-bw311          # or whatever holds mmdet
#   tmux new -s ablation
#   bash custom_configs/exp_configs/ablation/run_ablation_all.sh \
#        2>&1 | tee custom_configs/exp_configs/ablation/orchestrator.log
#
# Failures DON'T abort the sweep — they're logged and the next run starts.

set -o pipefail

# --- environment --------------------------------------------------------
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_ROOT="/home/pesquisador/pesquisa/filipe/vcnc_mil"
cd "$REPO_ROOT"

# Verify the conda env (or whichever Python) has mmdet importable.
if ! python -c "import mmdet" 2>/dev/null; then
    echo "[ABORT] python env sem mmdet. Ative o conda env antes (ex: conda activate mmlab-bw311)."
    exit 1
fi

PROGRESS_FILE="custom_configs/exp_configs/ablation/ablation_progress.txt"
WORK_BASE="work_dirs/ablation"
RUN_TIMEOUT="3h"

mkdir -p "$WORK_BASE"
if [ ! -s "$PROGRESS_FILE" ]; then
    echo "scenario,seed,status,mAP_final,mAP_best,wall_time_min" > "$PROGRESS_FILE"
fi

# --- run list (tier|scenario|seed) --------------------------------------
RUNS=(
    # TIER 1 (loc + class noise)
    "tier1|loc20_asym20|2025"
    "tier1|loc20_asym20|42"
    "tier1|loc20_mi20_bkg20_asym20|2025"
    "tier1|loc20_mi20_bkg20_asym20|42"
    "tier1|loc40_mi40_bkg40_sym40|2025"
    "tier1|loc40_mi40_bkg40_sym40|42"
    "tier1|loc40_mi40_bkg40_asym40|2025"
    "tier1|loc40_mi40_bkg40_asym40|42"
    "tier1|loc20_mi20_bkg20_sym40|2025"
    "tier1|loc20_mi20_bkg20_sym40|42"
    "tier1|loc20_mi20_bkg20_asym40|2025"
    "tier1|loc20_mi20_bkg20_asym40|42"
    "tier1|loc40_mi40_bkg40_sim20|2025"
    "tier1|loc40_mi40_bkg40_sim20|42"
    "tier1|loc40_mi40_bkg40_asym20|2025"
    "tier1|loc40_mi40_bkg40_asym20|42"
    # TIER 2
    "tier2|loc20_mi20_bkg20_sym20|2025"
    "tier2|loc20_mi20_bkg20_sym20|42"
    "tier2|sym40|2025"
    "tier2|sym40|42"
    "tier2|asym40|2025"
    "tier2|asym40|42"
    # TIER 3
    "tier3|sym20|2025"
    "tier3|sym20|42"
    "tier3|asym20|2025"
    "tier3|asym20|42"
    "tier3|sym60|2025"
    "tier3|sym60|42"
    "tier3|loc20|2025"
    "tier3|loc20|42"
    "tier3|loc60|2025"
    "tier3|loc60|42"
    # TIER 4
    "tier4|mi20|2025"
    "tier4|mi20|42"
    "tier4|bkg20|2025"
    "tier4|bkg20|42"
    "tier4|mi40|2025"
    "tier4|mi40|42"
    "tier4|bkg40|2025"
    "tier4|bkg40|42"
    "tier4|mi60|2025"
    "tier4|mi60|42"
    "tier4|bkg60|2025"
    "tier4|bkg60|42"
)
TOTAL=${#RUNS[@]}

# --- pre-validation: parse every final config via Config.fromfile ------
# Catches missing files, broken _base_ chains, dataset references that
# don't resolve, etc. ~5s total (single python invocation).
echo "[VALIDATION] Parsing $TOTAL configs via mmengine.Config.fromfile..."
{
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r tier scen seed <<< "$entry"
        echo "custom_configs/exp_configs/ablation/${tier}_${scen}/vcnc_oamil_seed${seed}.py"
    done
} | python -c "
import sys
from mmengine.config import Config
configs = [l.strip() for l in sys.stdin if l.strip()]
broken = []
for cfg in configs:
    try:
        Config.fromfile(cfg)
    except Exception as e:
        broken.append((cfg, type(e).__name__ + ': ' + str(e)[:240]))
if broken:
    print(f'[VALIDATION] {len(broken)} of {len(configs)} configs broken:', file=sys.stderr)
    for c, e in broken:
        print(f'  {c}\n      {e}', file=sys.stderr)
    sys.exit(1)
print(f'[VALIDATION] All {len(configs)} configs OK.')
"
if [ $? -ne 0 ]; then
    echo "[ABORT] Pre-validation failed — fix above and rerun. Sweep not started."
    exit 1
fi

# --- helpers ------------------------------------------------------------
extract_map() {
    # extract_map <log_file> <final|best>
    local log_file=$1 mode=$2 values
    values=$(grep -oE 'pascal_voc/mAP: [0-9]+\.[0-9]+' "$log_file" 2>/dev/null | awk '{print $2}')
    if [ -z "$values" ]; then
        echo "NA"
        return
    fi
    if [ "$mode" = "final" ]; then
        echo "$values" | tail -1
    else
        echo "$values" | sort -n | tail -1
    fi
}

run_one() {
    local tier=$1 scen=$2 seed=$3

    if grep -qE "^${scen},${seed},DONE," "$PROGRESS_FILE"; then
        echo "[SKIP] $scen seed=$seed (already DONE in $PROGRESS_FILE)"
        return 0
    fi

    local config="custom_configs/exp_configs/ablation/${tier}_${scen}/vcnc_oamil_seed${seed}.py"
    local work_dir="${WORK_BASE}/${tier}_${scen}_seed${seed}"
    local stdout_log="${work_dir}.stdout"

    mkdir -p "$work_dir"

    local start_ts
    start_ts=$(date +%s)
    echo "[STARTING] $scen seed=$seed at $(date '+%Y-%m-%d %H:%M:%S')"

    timeout "$RUN_TIMEOUT" python tools/train.py "$config" \
        --work-dir "$work_dir" \
        > "$stdout_log" 2>&1
    local exit_code=$?

    local end_ts elapsed_min
    end_ts=$(date +%s)
    elapsed_min=$(( (end_ts - start_ts) / 60 ))

    if [ $exit_code -eq 0 ]; then
        local log_file final best
        log_file=$(ls -1 "$work_dir"/*/[0-9]*.log 2>/dev/null | tail -1)
        if [ -n "$log_file" ]; then
            final=$(extract_map "$log_file" final)
            best=$(extract_map "$log_file" best)
        else
            final="NA"; best="NA"
        fi
        echo "[DONE] $scen seed=$seed in ${elapsed_min}m (mAP final=$final best=$best)"
        echo "${scen},${seed},DONE,${final},${best},${elapsed_min}" >> "$PROGRESS_FILE"
    elif [ $exit_code -eq 124 ]; then
        echo "[TIMEOUT] $scen seed=$seed after $RUN_TIMEOUT"
        echo "${scen},${seed},TIMEOUT,NA,NA,${elapsed_min}" >> "$PROGRESS_FILE"
    else
        echo "[FAILED] $scen seed=$seed exit=$exit_code in ${elapsed_min}m"
        echo "${scen},${seed},FAILED_exit${exit_code},NA,NA,${elapsed_min}" >> "$PROGRESS_FILE"
    fi
}

# --- main loop ----------------------------------------------------------
i=0
sweep_start=$(date +%s)
for entry in "${RUNS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r tier scen seed <<< "$entry"
    echo "==="
    echo "[$i/$TOTAL] $tier / $scen / seed=$seed"
    run_one "$tier" "$scen" "$seed"
done
sweep_end=$(date +%s)
sweep_min=$(( (sweep_end - sweep_start) / 60 ))

# --- summary ------------------------------------------------------------
echo "==="
echo "Sweep finished in ${sweep_min} minutes."
echo
echo "Final progress (last ${TOTAL} lines):"
tail -$((TOTAL + 1)) "$PROGRESS_FILE"
