#!/usr/bin/env bash
#
# setup_runpod.sh — one-command setup on a fresh RunPod pod (or any host).
# Installs Python deps and downloads the UMD dataset if missing. Idempotent:
# safe to re-run; skips anything already present (e.g. on a network volume).
#
#     git clone <repo> && cd cv-project
#     bash setup_runpod.sh
#     bash experiments/01_main_headline.sh
#
# (On RunPod, clone into /workspace so it lives on the persistent volume.)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== deps ==="
pip install -r requirements.txt

RAW="data/raw"
TOOLS_DIR="$RAW/part-affordance-dataset"
mkdir -p "$RAW"

if [ -d "$TOOLS_DIR/tools" ]; then
    echo "=== dataset already present ($TOOLS_DIR/tools) — skipping download ==="
else
    echo "=== downloading UMD Part Affordance (tools split) ==="
    wget -c https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz \
        -O "$RAW/tools.tar.gz"
    tar xzf "$RAW/tools.tar.gz" -C "$RAW"
    rm -f "$RAW/tools.tar.gz"
fi

# Clutter split (only needed for the robustness eval) — set CLUTTER=1 to fetch.
if [ "${CLUTTER:-0}" = "1" ] && [ ! -d "$RAW/part-affordance-clutter" ]; then
    echo "=== downloading UMD clutter split ==="
    wget -c https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-clutter.tar.gz \
        -O "$RAW/clutter.tar.gz"
    tar xzf "$RAW/clutter.tar.gz" -C "$RAW"
    rm -f "$RAW/clutter.tar.gz"
fi

echo "=== ready. next:  bash experiments/01_main_headline.sh ==="
