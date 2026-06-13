"""
diagnose_normals.py — Is the ~24-25 deg validation angular error a MODEL limit
or a GROUND-TRUTH limit?

READ-ONLY. Touches no checkpoints, no training state, writes only a few PNGs
to --out_dir. Safe to run while training is in progress.

It answers two questions on a random sample of the SAME val split training uses:

  (1) TRIVIAL-BASELINE TEST. How much better than a constant-normal predictor
      is the model? We measure the angular error (over affordance pixels) of:
        - "camera-facing"  : predict [0,0,1] everywhere
        - "per-image mean" : predict each image's own mean GT normal everywhere
                             (the best ANY constant-per-image predictor can do)
        - "global mean"    : predict the dataset-average normal everywhere
      If the model's 24-25 deg is barely below these, it has learned almost
      nothing beyond "guess a constant" -> something is capping it.

  (2) GT NOISE-FLOOR TEST. We blur each GT normal map (cv2.blur) and renormalize,
      then measure the angle between the raw GT normal and its own smoothed
      version over affordance pixels. This is the high-frequency JITTER baked
      into the targets by differentiating raw depth. A model CANNOT get below
      roughly this number however well it trains. If this self-inconsistency is
      ~15-20+ deg, raw-depth noise is the cap and depth smoothing is justified.

Interpretation cheat-sheet:
  - model_angle ~ per-image-mean baseline  -> model barely beat a constant.
  - GT jitter   ~ model_angle              -> targets are the floor (noise-capped).
  - GT jitter   << model_angle             -> targets are clean-ish; the model
                                              still has room to improve, so DON'T
                                              touch the GT, just let it train.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_TOOLS, TRAIN_INTRINSICS  # noqa: E402
from utils.dataset import UMDAffordanceDataset, make_split  # noqa: E402


def _angles_deg(pred_hw3, gt_hw3, active_hw):
    """Per-pixel angle (deg) between two unit-normal maps, over active pixels."""
    cos = (pred_hw3 * gt_hw3).sum(axis=-1)
    cos = np.clip(cos[active_hw], -1.0 + 1e-6, 1.0 - 1e-6)
    return np.degrees(np.arccos(cos))


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + 1e-9)


def get_args():
    p = argparse.ArgumentParser("Diagnose normal angular-error floor")
    p.add_argument("--split_type", default="novel_instance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_file", default=None)
    p.add_argument("--n_samples", type=int, default=200,
                   help="random val images to evaluate")
    p.add_argument("--n_vis", type=int, default=4,
                   help="GT-normal panels to save for eyeballing")
    p.add_argument("--blur", type=int, default=5,
                   help="box-blur kernel for the GT noise-floor estimate")
    p.add_argument("--history", default=str(PROJECT_ROOT / "runs_eccv/main/history.jsonl"),
                   help="read the model's last val angle for context (optional)")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "runs_eccv/normal_diagnostic"))
    return p.parse_args()


def model_angle_from_history(path):
    try:
        last = None
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("phase") == "val" and "angle_deg" in rec:
                    last = rec["angle_deg"]
        return last
    except (OSError, json.JSONDecodeError):
        return None


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_ds = UMDAffordanceDataset(raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
                                  augment=False)
    _, val_idx = make_split(val_ds, split_type=args.split_type, seed=args.seed,
                            val_frac=args.val_frac, split_file=args.split_file)

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(val_idx))
    pick = rng.choice(np.asarray(val_idx), size=n, replace=False)
    print(f"val images={len(val_idx)} | sampling {n} | split={args.split_type}")

    # Accumulators: (sum_of_degrees, pixel_count) so the mean is area-weighted
    # exactly like the training metric (which means over affordance pixels).
    acc = {k: [0.0, 0] for k in
           ("camera_facing", "per_image_mean", "gt_jitter")}
    global_sum = np.zeros(3, dtype=np.float64)

    per_image_mean_vecs = []  # store to compute global-mean error in pass 2
    actives = []

    n_vis_done = 0
    for s_i, idx in enumerate(pick):
        item = val_ds[int(idx)]
        nrm = item["normals"].numpy().transpose(1, 2, 0)        # [H,W,3]
        msk = item["mask"].numpy()                               # [C,H,W]
        active = msk.sum(axis=0) > 0                             # [H,W]
        # Drop invalid-depth pixels (compute_normals zeroes them -> norm 0).
        active &= np.linalg.norm(nrm, axis=-1) > 0.5
        if active.sum() == 0:
            per_image_mean_vecs.append(None)
            actives.append(None)
            continue
        nrm = _unit(nrm)

        # (1a) camera-facing constant [0,0,1] -> cos == n_z
        cf = _angles_deg(np.broadcast_to(np.array([0, 0, 1.0]), nrm.shape),
                         nrm, active)
        acc["camera_facing"][0] += cf.sum(); acc["camera_facing"][1] += cf.size

        # (1b) per-image mean constant (best possible single vector for this img)
        mean_vec = _unit(nrm[active].mean(axis=0))
        pim = _angles_deg(np.broadcast_to(mean_vec, nrm.shape), nrm, active)
        acc["per_image_mean"][0] += pim.sum(); acc["per_image_mean"][1] += pim.size
        per_image_mean_vecs.append(mean_vec)
        actives.append(active)
        global_sum += nrm[active].sum(axis=0)

        # (2) GT noise floor: raw GT vs its own blurred+renormalized version
        sm = cv2.blur(nrm, (args.blur, args.blur))
        sm = _unit(sm)
        jit = _angles_deg(sm, nrm, active)
        acc["gt_jitter"][0] += jit.sum(); acc["gt_jitter"][1] += jit.size

        # Save a few panels to eyeball
        if n_vis_done < args.n_vis:
            _save_panel(out_dir, n_vis_done, item, nrm, active)
            n_vis_done += 1

    # Global-mean constant (pass 2)
    g_vec = _unit(global_sum)
    gm_sum, gm_cnt = 0.0, 0
    for idx, mv, active in zip(pick, per_image_mean_vecs, actives):
        if active is None:
            continue
        item = val_ds[int(idx)]
        nrm = _unit(item["normals"].numpy().transpose(1, 2, 0))
        gm = _angles_deg(np.broadcast_to(g_vec, nrm.shape), nrm, active)
        gm_sum += gm.sum(); gm_cnt += gm.size

    def mean(pair):
        return pair[0] / pair[1] if pair[1] else float("nan")

    model_ang = model_angle_from_history(args.history)

    print("\n================ NORMAL ANGULAR-ERROR DIAGNOSTIC ================")
    if model_ang is not None:
        print(f"  MODEL (last val angle_deg)        : {model_ang:6.2f} deg")
    print( "  --- trivial constant baselines (lower = easier to beat) ---")
    print(f"  camera-facing [0,0,1] everywhere  : {mean(acc['camera_facing']):6.2f} deg")
    print(f"  global mean normal everywhere     : {gm_sum / gm_cnt if gm_cnt else float('nan'):6.2f} deg")
    print(f"  per-image mean (best constant)    : {mean(acc['per_image_mean']):6.2f} deg")
    print( "  --- GT noise floor ---")
    print(f"  GT jitter (raw vs blur{args.blur}, renorm) : {mean(acc['gt_jitter']):6.2f} deg")
    print("================================================================")
    print("READ: if MODEL ~ per-image-mean, it barely beat a constant.")
    print("      if GT jitter ~ MODEL, the targets are the floor (noise-capped).")
    print("      if GT jitter << MODEL, targets are clean -> don't touch GT, keep training.")
    print(f"\nPanels saved to: {out_dir}")


def _save_panel(out_dir, i, item, nrm, active):
    """RGB | GT-normal (RGB-coded) | affordance overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # De-normalize the input RGB for display.
    rgb = item["rgb"].numpy().transpose(1, 2, 0)
    rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    rgb = np.clip(rgb, 0, 1)
    nvis = (nrm + 1.0) / 2.0
    nvis[~active] = 0
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb); ax[0].set_title(item.get("tool_name", "")); ax[0].axis("off")
    ax[1].imshow(nvis); ax[1].set_title("GT normals (affordance px)"); ax[1].axis("off")
    ov = rgb.copy(); ov[active] = 0.5 * ov[active] + 0.5 * nvis[active]
    ax[2].imshow(ov); ax[2].set_title("overlay"); ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"gt_normals_{i:02d}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
