"""
===========================================================
Ellipse Fitting Demo (robust + plot, NumPy-only)
===========================================================

Steps:
  1) Load a CSV grid
  2) Preprocess (binarize + clean)
  3) Extract contour points
  4) Fit with RANSAC + optional LM refinement
  5) Plot the result

Author
------
Adrian Utge Le Gall, 2025
"""

# --- Imports --------------------------------------------------------------
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from elli_fit import load_binary_matrix
from elli_fit.core import (
    preprocess_mask,
    edge_points,
    edge_points_ordered,
    fit_ellipse_ransac,
    ellipse_points_image,
    refine_ellipse_lm,
    align_to_x_axis_image,
)

# --- CLI -----------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ellipse fitting demo.")
    p.add_argument("csv", help="Path to the CSV grid.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-clear-border", action="store_true")
    p.add_argument("--min-neighbors", type=int, default=2)
    p.add_argument("--ransac-iters", type=int, default=800)
    p.add_argument("--ransac-sample", type=int, default=7)
    p.add_argument("--ransac-thresh", type=float, default=0.25)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--save", type=str, default="")
    p.add_argument("--title", type=str, default="Ellipse Fit Overlay")
    p.add_argument("--show-raw", action="store_true")

    # Alignment / mirroring controls
    p.add_argument("--align-x", action="store_true",
                   help="Rotate experimental edge points so the major axis is horizontal before plotting.")
    p.add_argument("--mirror", action="store_true",
                   help="Mirror ONLY the fitted ellipse across the vertical axis through its center.")
    return p.parse_args(argv)

# --- Main ----------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    # Load + preprocess
    M = load_binary_matrix(args.csv)
    print(f"[info] loaded {M.shape} from: {args.csv}")
    B = preprocess_mask(
        M,
        threshold=args.threshold,
        clear_border=(not args.no_clear_border),
        min_neighbors=max(0, args.min_neighbors),
    )

    # Extract contour (ordered preferred, fallback to unordered)
    x, y = edge_points_ordered(B)
    if len(x) < 20:
        print("[warn] ordered contour too short; falling back to unordered edges")
        x, y = edge_points(B, threshold=args.threshold)
    if len(x) < 6:
        print("[error] not enough edge points for ellipse fitting.")
        return 1

    # --- Step 1: robust fit (RANSAC) -------------------------------------
    cx, cy, a, b, th = fit_ellipse_ransac(
        x, y,
        iters=args.ransac_iters,
        sample_size=args.ransac_sample,
        inlier_thresh=args.ransac_thresh,
        min_inliers=max(20, len(x)//3),
    )
    print(f"[ransac] a={a:.2f}, b={b:.2f}, θ={np.degrees(th):.1f}°")

    # --- Step 2: optional LM refinement ----------------------------------
    try:
        cx, cy, a, b, th = refine_ellipse_lm(x, y, cx, cy, a, b, th, iters=30, lam=1e-2)
        print(f"[refined] a={a:.2f}, b={b:.2f}, θ={np.degrees(th):.1f}°")
    except Exception as e:
        print(f"[warn] refinement skipped: {e}")

    # --- Step 3: align display frame if requested ------------------------
    if args.align_x:
        # rotate experimental edge points to make major axis horizontal
        x_disp, y_disp, cx_disp, cy_disp, a_disp, b_disp, th_disp = align_to_x_axis_image(
            x, y, cx, cy, a, b, th
        )
    else:
        x_disp, y_disp, cx_disp, cy_disp, a_disp, b_disp, th_disp = x, y, cx, cy, a, b, th

    # Sample the fitted ellipse in IMAGE coords (y downward) for overlay
    Xf, Yf = ellipse_points_image(cx_disp, cy_disp, a_disp, b_disp, th_disp, n=args.samples)

    # Mirror ONLY the fitted curve if requested (across x = cx_disp)
    if args.mirror:
        Xf = 2.0 * cx_disp - Xf
        th_disp = -th_disp  # purely cosmetic in the legend

    # --- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.set_facecolor("white")

    # Background image: skip it when aligning (since the image is not rotated)
    if not args.align_x:
        img = M if args.show_raw else B
        ax.imshow(img, cmap="Greys", origin="upper", vmin=0, vmax=1)

    ax.scatter(x_disp, y_disp, s=8, c="dodgerblue", label=f"edge points (n={len(x_disp)})")
    ax.plot(Xf, Yf, "r", lw=2, label=f"fit θ={np.degrees(th_disp):.1f}° (aligned={args.align_x})")

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(frameon=False, loc="best")
    ax.set_title(args.title)
    fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"[ok] saved figure -> {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
