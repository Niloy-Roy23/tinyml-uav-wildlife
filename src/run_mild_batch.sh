#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1

PY="./.venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="reports/mild_batch_${STAMP}"
STATUS_FILE="${LOG_DIR}/status.txt"

if [[ ! -f "$PY" ]]; then
    echo "ERROR: Python environment not found: $PY"
    exit 1
fi

# Refuse to overwrite any existing mild experiment outputs.
outputs=(
    models/improved_cnn_mild.keras
    models/improved_cnn_mild_final.keras
    reports/improved_mild_training_history.csv
    reports/improved_mild_training_history.json
    models/rotor_augmentation_cnn_mild.keras
    models/rotor_augmentation_cnn_mild_best.keras
    reports/rotor_augmentation_mild_history.csv
    models/rotor_consistent_cnn_mild.keras
    models/rotor_consistent_cnn_mild_best.keras
    reports/rotor_consistent_mild_history.csv
    models/mfcc_cnn1d_mild.keras
    models/mfcc_cnn1d_mild_best.keras
    models/mfcc_cnn1d_mild_normalization.npz
    models/mfcc_mlp_v2_mild.keras
    models/mfcc_mlp_v2_mild_best.keras
    models/mfcc_normalization_v2_mild.npz
    models/mfcc_mlp_mild.keras
    models/mfcc_mlp_mild_best.keras
    models/mfcc_normalization_mild.npz
)

existing=()
for file in "${outputs[@]}"; do
    [[ -e "$file" ]] && existing+=("$file")
done

if (( ${#existing[@]} > 0 )); then
    echo "STOPPED: Existing output files found. Nothing was trained."
    printf '  %s\n' "${existing[@]}"
    echo "Inspect these files before deciding whether to rerun."
    exit 2
fi

mkdir -p "$LOG_DIR"
echo "Mild-weight training batch: $STAMP" | tee "$STATUS_FILE"
echo "Weights: bird=0.875, insect=1.25, background=1.0" | tee -a "$STATUS_FILE"

scripts=(
    src/train_improved_mild.py
    src/train_rotor_augmentation_mild.py
    src/train_rotor_consistent_mild.py
    src/train_mfcc_cnn1d_mild.py
    src/train_mfcc_mlp_v2_mild.py
    src/train_mfcc_model_mild.py
)

failed=0

for script in "${scripts[@]}"; do
    name="$(basename "$script" .py)"
    log="${LOG_DIR}/${name}.log"

    echo
    echo "========================================"
    echo "Starting: $script"
    echo "Log: $log"

    if "$PY" "$script" > "$log" 2>&1; then
        echo "PASS: $script" | tee -a "$STATUS_FILE"
    else
        code=$?
        echo "FAIL (exit ${code}): $script — see $log" | tee -a "$STATUS_FILE"
        failed=$((failed + 1))
    fi
done

echo
echo "========================================"
echo "Batch complete. Failed scripts: $failed"
echo "Status: $STATUS_FILE"
echo "Logs: $LOG_DIR"
cat "$STATUS_FILE"

if (( failed > 0 )); then
    exit 1
fi
