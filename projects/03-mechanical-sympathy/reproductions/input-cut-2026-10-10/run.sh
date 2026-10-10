#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SAMUDRA_ROOT:?Set SAMUDRA_ROOT to the Samudra checkout}"
: "${CHECKPOINT:?Set CHECKPOINT to the one-degree EMA checkpoint}"
: "${PREPARED_CASE:?Set PREPARED_CASE to the copied eval_case.pt}"
: "${NEURON_PYTHON:?Set NEURON_PYTHON to the Neuron environment Python}"
report_path="${REPORT_PATH:-${bundle_dir}/reproduced_inputcut.json}"
# Select an available logical core. CPU_AFFINITY is machine-specific.
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}"
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_CC_FLAGS='--target=trn2 --auto-cast=matmult --auto-cast-type=bf16 --logical-nc-config=2'
export PYTHONUNBUFFERED=1
launch=()
if [[ -n "${CPU_AFFINITY:-}" ]]; then launch=(taskset -c "$CPU_AFFINITY"); fi
"${launch[@]}" "$NEURON_PYTHON" "$bundle_dir/benchmark_forward.py" \
  --samudra-root "$SAMUDRA_ROOT" "$bundle_dir/eval.replay.yaml" \
  --checkpoint "$CHECKPOINT" --case "$PREPARED_CASE" \
  --backend neuron --threads 8 --warmup 3 --single-repeats 20 \
  --tile-rows 30 --tile-min-pixels 64800 --fold-batch-norm --input-cut \
  --output "$report_path"
