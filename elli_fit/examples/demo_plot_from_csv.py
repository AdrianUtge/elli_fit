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
    _rotate_points_image,
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
    p.add_argument("--align-x", action="store_true",
                   help="Rotate data so the experimental ellipse major axis is horizontal before plotting.")
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

    # Extract contour
    x, y = edge_points_ordered(B)
    y = - y
    if len(x) < 20:
        print("[warn] ordered contour too short; falling back to unordered edges")
        x, y = edge_points(B, threshold=args.threshold)
    if len(x) < 6:
        print("[error] not enough edge points for ellipse fitting.")
        return 1

    # --- Step 1: RANSAC fit ---------------------------------------------
    cx, cy, a, b, th = fit_ellipse_ransac(
        x, y,
        iters=args.ransac_iters,
        sample_size=args.ransac_sample,
        inlier_thresh=args.ransac_thresh,
        min_inliers=max(20, len(x)//3),
    )
    print(f"[ransac] a={a:.2f}, b={b:.2f}, θ={np.degrees(th):.1f}°")

    # --- Step 2: geometric refinement (optional) -------------------------
    try:
        cx, cy, a, b, th = refine_ellipse_lm(
            x, y, cx, cy, a, b, th, iters=30, lam=1e-2)
        print(f"[refined] a={a:.2f}, b={b:.2f}, θ={np.degrees(th):.1f}°")
    except Exception as e:
        print(f"[warn] refinement skipped: {e}")

    # --- Step 3: allignment under the same axis -------------------------
       # After: cx, cy, a, b, th computed
    if args.align_x:
        # Rotate experimental edge points so the major axis becomes horizontal
        x_disp, y_disp, cx_disp, cy_disp, a_disp, b_disp, th_disp = align_to_x_axis_image(
            x, y, cx, cy, a, b, th
        )
    else:
        # No alignment: display in image coords as-is
        x_disp, y_disp, cx_disp, cy_disp, a_disp, b_disp, th_disp = x, y, cx, cy, a, b, th

    # Sample ellipse in IMAGE coords using the display params
    Xf, Yf = ellipse_points_image(
        cx_disp, cy_disp, a_disp, b_disp, th_disp, n=args.samples)

    # Points de l’ellipse (en repère image)

    Xf, Yf = ellipse_points_image(
        cx_disp, cy_disp, a_disp, b_disp, th_disp, n=args.samples)
    
    
    
    
    
    
    
        # points de la courbe ellipse (repère image)
    Xf, Yf = ellipse_points_image(cx_disp, cy_disp, a_disp, b_disp, th_disp, n=args.samples)
    
    # --- MIRROR only the fitted curve (red) about the vertical axis x = cx_disp
    if args.mirror:
        Xf = 2.0 * cx_disp - Xf  # y unchanged
        # (facultatif) l’angle affiché devient -θ côté image
        th_disp = -th_disp
    
    
    
    
    
    
    
    # --- Plot (image coords cohérents) ---------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.set_facecolor("white")

    # Showing the raw image when aligned is visually confusing (it remains unrotated).
    # Either skip it or keep it only when not aligning:
    if not args.align_x:
        img = M if args.show_raw else B
        ax.imshow(img, cmap="Greys", origin="upper", vmin=0, vmax=1)

    # Plot the (possibly rotated) experimental points
    ax.scatter(x_disp, y_disp, s=8, c="dodgerblue",
               label=f"edge points (n={len(x_disp)})")

    # Plot the ellipse that matches the display frame
    ax.plot(Xf, Yf, "r", lw=2,
            label=f"fit θ={np.degrees(th_disp):.1f}° (aligned={args.align_x})")

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"[ok] saved figure -> {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
