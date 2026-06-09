#!/usr/bin/env bash
#
# archive_run.sh — Snapshot the active checkpoints/ directory into runs/
#
# Usage:
#     bash scripts/archive_run.sh <descriptor>
#
# Produces:
#     runs/<descriptor>_YYYY-MM/
#       best.pth
#       last.pth
#       history.jsonl
#       evaluation_val.json   (if present)
#       run_config.json
#       class_pixel_counts.json   (if present)
#       samples/   (if present)
#       training_curves.png   (if present)
#       training_summary.txt   (if present)
#
# This is a copy, not a move: checkpoints/ remains the active working
# directory and continues to serve scripts/{train,evaluate,visualize}.py.
# Re-archive whenever a run reaches a natural milestone.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/archive_run.sh <descriptor>" >&2
    echo "Example: bash scripts/archive_run.sh multiclass_weighted" >&2
    exit 1
fi

DESCRIPTOR="$1"
DATE_SUFFIX="$(date +%Y-%m)"
SRC_DIR="checkpoints"
DEST_DIR="runs/${DESCRIPTOR}_${DATE_SUFFIX}"

if [ ! -d "$SRC_DIR" ]; then
    echo "Source $SRC_DIR not found. Run from the project root." >&2
    exit 1
fi

if [ -e "$DEST_DIR" ]; then
    echo "Destination $DEST_DIR already exists." >&2
    echo "Either delete it first, or use a different descriptor." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

# Required artifacts
for f in best.pth last.pth history.jsonl run_config.json; do
    if [ -f "$SRC_DIR/$f" ]; then
        cp "$SRC_DIR/$f" "$DEST_DIR/$f"
        echo "  copied  $f"
    else
        echo "  missing $f (skipping)"
    fi
done

# Optional artifacts
for f in evaluation_val.json evaluation_train.json evaluation_all.json \
         class_pixel_counts.json training_curves.png training_summary.txt; do
    if [ -f "$SRC_DIR/$f" ]; then
        cp "$SRC_DIR/$f" "$DEST_DIR/$f"
        echo "  copied  $f"
    fi
done

# Sample directory (one PNG per sample) is optional
if [ -d "$SRC_DIR/samples" ]; then
    cp -R "$SRC_DIR/samples" "$DEST_DIR/samples"
    echo "  copied  samples/"
fi

echo ""
echo "Archived $SRC_DIR -> $DEST_DIR"
echo ""
echo "Next:"
echo "  1. Append a row to runs/INDEX.md describing this run."
echo "  2. If the run produced a notable finding, add a subsection to docs/RESULTS.md."
echo "  3. Add a one-paragraph entry to docs/CHANGELOG.md."
