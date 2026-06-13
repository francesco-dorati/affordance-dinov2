#!/usr/bin/env bash
#
# 04_ablation_no_class_weights.sh — ABLATION: effect of frequency-inverse
# per-class loss weights.
#
# Same architecture and split, but --no-class_weights. Compare this run's
# per-class weighted F-measure against 01 (main) to show the class-weighting
# contribution, especially on rare classes (support / scoop / pound).
# Flag-only ablation.
#
# RESUMABLE: re-run to continue from <OUT>/last.pth.
#
#     bash experiments/04_ablation_no_class_weights.sh

set -euo pipefail
cd "$(dirname "$0")/.."          # project root

# Local conda env if present; on RunPod/other hosts use the env from setup_runpod.sh.
if command -v conda >/dev/null 2>&1; then conda activate robotics-affordance 2>/dev/null || true; fi

OUT="runs_eccv/ablation_no_class_weights"
mkdir -p "$OUT"

echo "=== [04] ABLATION no-class-weights -> $OUT ==="
python scripts/train.py \
    --epochs 40 \
    --batch_size 8 \
    --split_type novel_instance \
    --no-class_weights \
    --output_dir "$OUT" \
    --resume

echo "=== [04] Final evaluation (full val, weighted F-measure) ==="
python scripts/evaluate.py \
    --checkpoint "$OUT/best_wfb.pth" \
    --split_type novel_instance \
    --output_dir "$OUT"
