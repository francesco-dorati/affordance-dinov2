#!/usr/bin/env bash
#
# 02_deeplab_baseline.sh — DeepLabv3-ResNet50 baseline (paper comparison control).
#
# Standard off-the-shelf segmentation model on the SAME split + SAME metric.
# Validates the F-measure implementation (should land near AffordanceNet
# Table II's DeepLab = 0.733) and gives the in-house apples-to-apples row.
# Self-evaluates at the end -> <OUT>/evaluation_baseline.json (evaluate.py is
# for the DINOv2 model only, so the baseline scores itself).
#
# RESUMABLE: re-run to continue from <OUT>/last.pth.
# NOTE: ResNet-50 full fine-tune is memory-heavy. --batch_size 4 is a safe
# default for a 12 GB 5070; bump to 8 if it fits.
#
#     bash experiments/02_deeplab_baseline.sh

set -euo pipefail
cd "$(dirname "$0")/.."          # project root

# Local conda env if present; on RunPod/other hosts use the env from setup_runpod.sh.
if command -v conda >/dev/null 2>&1; then conda activate robotics-affordance 2>/dev/null || true; fi

OUT="runs_eccv/baseline_deeplab"
mkdir -p "$OUT"

echo "=== [02] DeepLabv3 baseline -> $OUT ==="
python scripts/train_baseline.py \
    --epochs 40 \
    --batch_size 4 \
    --split_type novel_instance \
    --output_dir "$OUT" \
    --resume
