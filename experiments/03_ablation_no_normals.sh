#!/usr/bin/env bash
#
# 03_ablation_no_normals.sh — ABLATION: does the multi-task (surface-normal)
# head help the affordance masks?
#
# Same architecture, but the normal losses are zeroed (--w_normal 0 --w_smooth 0)
# so the shared trunk gets no normal supervision. Compare this run's mask
# weighted F-measure against 01 (main) to isolate the multi-task contribution.
# Flag-only ablation — no code change, no separate model.
#
# RESUMABLE: re-run to continue from <OUT>/last.pth.
#
#     bash experiments/03_ablation_no_normals.sh

set -euo pipefail
cd "$(dirname "$0")/.."          # project root

# Local conda env if present; on RunPod/other hosts use the env from setup_runpod.sh.
if command -v conda >/dev/null 2>&1; then conda activate robotics-affordance 2>/dev/null || true; fi

OUT="runs_eccv/ablation_no_normals"
mkdir -p "$OUT"

echo "=== [03] ABLATION no-normals (w_normal=0, w_smooth=0) -> $OUT ==="
python scripts/train.py \
    --epochs 40 \
    --batch_size 8 \
    --split_type novel_instance \
    --w_normal 0 \
    --w_smooth 0 \
    --output_dir "$OUT" \
    --resume

echo "=== [03] Final evaluation (full val, weighted F-measure) ==="
python scripts/evaluate.py \
    --checkpoint "$OUT/best_wfb.pth" \
    --split_type novel_instance \
    --output_dir "$OUT"
