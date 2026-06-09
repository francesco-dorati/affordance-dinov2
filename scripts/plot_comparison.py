"""
plot_comparison.py — Build the binary-vs-multi-class comparison figure.

Reads two evaluation_val.json files (e.g. the binary baseline and the final
weighted multi-class run), produces a single PNG with grouped horizontal bars
per tool plus an overall-mean summary panel. This is the headline results
figure for the presentation.

Caveats baked into the layout:
  - For tools where one report has no IoU entry (older binary report may
    omit class breakdown), the bar is drawn at NaN -> not shown.
  - The overall IoU@0.5 numbers are NOT strictly apples-to-apples (binary is
    2-class; multi-class is mean across 7 channels). The script annotates
    this in the figure caption so the slide audience reads it correctly.

Usage:
    python scripts/plot_comparison.py \
        --baseline checkpoints_binary/evaluation_val.json \
        --new      checkpoints/evaluation_val.json \
        --baseline_label "Binary (grasp + wrap-grasp)" \
        --new_label      "Multi-class (7) + class weights" \
        --output reports/binary_vs_multiclass.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_tool_iou(report: dict, tool: str, key: str = "iou@0.5") -> float:
    rec = report.get("per_tool", {}).get(tool)
    if rec is None:
        return float("nan")
    v = rec.get(key)
    return float(v) if v is not None else float("nan")


def get_overall_iou(report: dict, key: str = "iou@0.5") -> float:
    return float(report["overall"][key])


def main():
    p = argparse.ArgumentParser("Comparison plot for two evaluation reports")
    p.add_argument("--baseline", required=True,
                   help="Path to the 'before' evaluation_val.json")
    p.add_argument("--new", required=True,
                   help="Path to the 'after' evaluation_val.json")
    p.add_argument("--baseline_label", default="Baseline")
    p.add_argument("--new_label",      default="Final")
    p.add_argument("--output", default="reports/binary_vs_multiclass.png")
    p.add_argument("--sort_by", default="delta",
                   choices=["delta", "name", "baseline", "new"],
                   help="How to order tools on the y-axis.")
    p.add_argument("--metric", default="iou@0.5",
                   help="Which per-tool metric to plot (default iou@0.5)")
    args = p.parse_args()

    baseline = load_report(args.baseline)
    new      = load_report(args.new)

    tools = sorted(set(baseline.get("per_tool", {})) |
                   set(new.get("per_tool", {})))
    if not tools:
        raise SystemExit("No tools found in either report.")

    rows = []
    for t in tools:
        b = get_tool_iou(baseline, t, args.metric)
        n = get_tool_iou(new,      t, args.metric)
        d = (n - b) if (not np.isnan(b) and not np.isnan(n)) else float("nan")
        rows.append((t, b, n, d))

    if args.sort_by == "delta":
        rows.sort(key=lambda r: (np.isnan(r[3]), -(r[3] if not np.isnan(r[3]) else 0)))
    elif args.sort_by == "name":
        rows.sort(key=lambda r: r[0])
    elif args.sort_by == "baseline":
        rows.sort(key=lambda r: -(r[1] if not np.isnan(r[1]) else -1))
    elif args.sort_by == "new":
        rows.sort(key=lambda r: -(r[2] if not np.isnan(r[2]) else -1))

    names   = [r[0] for r in rows]
    b_vals  = [r[1] for r in rows]
    n_vals  = [r[2] for r in rows]
    deltas  = [r[3] for r in rows]

    # ---------- Figure layout: 3-to-1 width split ----------
    fig = plt.figure(figsize=(14, max(6, 0.45 * len(names) + 2)))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)

    # ---------- Left: per-tool grouped bars ----------
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(names))
    h = 0.4
    ax.barh(y - h/2, [v if not np.isnan(v) else 0 for v in b_vals], h,
            label=args.baseline_label,
            color="#9aa0a6", edgecolor="black", linewidth=0.4)
    ax.barh(y + h/2, [v if not np.isnan(v) else 0 for v in n_vals], h,
            label=args.new_label,
            color="#1a73e8", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{args.metric}")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-tool IoU comparison", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # Annotate deltas to the right of the longer bar
    for i, (b, n, d) in enumerate(zip(b_vals, n_vals, deltas)):
        if np.isnan(d):
            txt = "n/a"
            color = "tab:gray"
        else:
            sign = "+" if d >= 0 else ""
            txt = f"{sign}{d:.2f}"
            color = ("tab:green" if d >= 0.02
                     else "tab:red" if d <= -0.02 else "tab:gray")
        anchor = max(
            b if not np.isnan(b) else 0.0,
            n if not np.isnan(n) else 0.0,
        )
        ax.text(anchor + 0.015, i, txt,
                va="center", ha="left", fontsize=8, color=color)

    # ---------- Right: overall summary ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ob = get_overall_iou(baseline, args.metric)
    on = get_overall_iou(new,      args.metric)
    x = np.arange(2)
    ax2.bar(x, [ob, on],
            color=["#9aa0a6", "#1a73e8"],
            edgecolor="black", linewidth=0.4, width=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels([args.baseline_label, args.new_label],
                        rotation=15, fontsize=9, ha="right")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel(args.metric)
    ax2.set_title("Overall (mean) IoU", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, [ob, on]):
        ax2.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    delta_overall = on - ob
    ax2.text(0.5, -0.18,
             f"Δ = {delta_overall:+.3f}",
             transform=ax2.transAxes, ha="center", fontsize=10,
             color=("tab:green" if delta_overall > 0 else "tab:red"))

    # ---------- Figure title + caveat ----------
    fig.suptitle(
        f"{args.baseline_label} → {args.new_label}",
        fontsize=14, y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Note: 'binary' overall IoU is computed over a 2-class union; "
        "'multi-class' overall IoU is the mean across 7 affordance channels. "
        "The Δ underestimates the qualitative gain.",
        ha="center", fontsize=8, color="#444",
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote comparison plot to {out_path}")

    # ---------- Console summary ----------
    print("\nPer-tool deltas (sorted by improvement):")
    for t, b, n, d in sorted(rows, key=lambda r: -(r[3] if not np.isnan(r[3]) else -1)):
        b_s = "n/a" if np.isnan(b) else f"{b:.3f}"
        n_s = "n/a" if np.isnan(n) else f"{n:.3f}"
        d_s = "n/a" if np.isnan(d) else f"{d:+.3f}"
        print(f"  {t:15s}  {args.baseline_label[:18]:18s} {b_s}  →  "
              f"{args.new_label[:18]:18s} {n_s}  (Δ {d_s})")


if __name__ == "__main__":
    main()
