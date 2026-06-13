#!/usr/bin/env bash
#
# 01_main_headline.sh — THE headline run for the ECCV workshop paper.
#
# Multi-task affordance + surface-normal model on the canonical Myers
# novel-instance split, with the normal-rotation fix and class weights.
# Produces the number that goes in the comparison table (weighted F-measure).
#
# RESUMABLE: re-run this exact script after a GPU interruption — it continues
# from <OUT>/last.pth instead of restarting. Safe to run as many times as the
# 5070 frees up.
#
# Run from anywhere:
#     bash experiments/01_main_headline.sh

set -euo pipefail
cd "$(dirname "$0")/.."          # project root

# Local conda env if present; on RunPod/other hosts use the env from setup_runpod.sh.
if command -v conda >/dev/null 2>&1; then conda activate robotics-affordance 2>/dev/null || true; fi

OUT="runs_eccv/main"
mkdir -p "$OUT"

echo "=== [01] MAIN headline run -> $OUT ==="
python scripts/train.py \
    --epochs 40 \
    --batch_size 8 \
    --split_type novel_instance \
    --output_dir "$OUT" \
    --resume

echo "=== [01] Final evaluation (full val, weighted F-measure) ==="
python scripts/evaluate.py \
    --checkpoint "$OUT/best_wfb.pth" \
    --split_type novel_instance \
    --output_dir "$OUT"
