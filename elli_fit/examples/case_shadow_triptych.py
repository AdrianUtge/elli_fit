"""
===========================================================
Practical Case — Two-shadow diameter + ellipse fit (triptych)
===========================================================

Purpose
-------
Process a single grayscale image containing 3 vertical panels
(same experiment, 10 ms apart). For each panel:
  1) Detect the two darkest shadows (largest blobs)
  2) Measure horizontal separation in pixels and convert to mm
     - either with --mm-per-px
     - or with a linear CSV calibration sep_px,diam_mm
  3) Build a centered ROI between the shadows
  4) Extract edges, then fit a robust ellipse with `elli_fit`
  5) Save an overlay PNG per panel and a summary CSV

Library
-------
- Uses your existing `elli_fit` library as-is (no changes required)
  Functions: preprocess_mask, edge_points(_ordered), fit_ellipse_ransac,
             refine_ellipse_lm (optional), ellipse_points_image

Usage
-----
  python3 examples/case_shadow_triptych.py IMG.png \
      --mm-per-px 0.045 --refine --save-dir out_triptych

  # or with a CSV calibration (two cols: sep_px,diam_mm)
  python3 examples/case_shadow_triptych.py IMG.png \
      --calib-csv calib.csv --refine --save-dir out_triptych

Notes
-----
- If panels are not strictly equal widths, you can still use --panels N
  (equal split). For custom borders, extend `split_equal_panels(...)`.

Author
------
Adrian Utge Le Gall, 2025
"""

# --- Imports --------------------------------------------------------------

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from elli_fit.core import (
    preprocess_mask,
    edge_points,
    edge_points_ordered,
    fit_ellipse_ransac,
    refine_ellipse_lm,
    ellipse_points_image,
)


# --- I/O (image) ----------------------------------------------------------

def imread_gray(path: Path) -> np.ndarray:
    """
    Read PNG/JPG as float grayscale in [0, 1] using matplotlib (no extra deps).
    """
    arr = plt.imread(str(path))
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # drop alpha if present
            arr = arr[..., :3]
        # perceptual luminance
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    arr = arr.astype(float)
    if arr.max() > 1.5:  # handle [0..255]
        arr /= 255.0
    return arr


# --- Thresholding (Otsu fallback) -----------------------------------------

def otsu_thresh01(gray01: np.ndarray) -> float:
    """
    Otsu threshold on [0,1] grayscale → returns scalar threshold in [0,1].
    """
    g = np.clip(gray01, 0.0, 1.0)
    hist, _ = np.histogram(g.ravel(), bins=256, range=(0, 1))
    p = hist.astype(float) / max(1, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.linspace(0, 1, 256))
    mu_t = mu[-1]
    # between-class variance
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return (k + 0.5) / 256.0


# --- Connected components (largest two blobs, 4-neighborhood) -------------

def largest_two_components(B: np.ndarray):
    """
    Return the two largest connected components of a binary mask (bool).
    Pure NumPy BFS (4-neighborhood). Returns (mask1, mask2).
    """
    H, W = B.shape
    visited = np.zeros_like(B, dtype=bool)
    blobs = []

    for i in range(H):
        for j in range(W):
            if B[i, j] and not visited[i, j]:
                q = [(i, j)]
                visited[i, j] = True
                coords = [(i, j)]
                while q:
                    ci, cj = q.pop()
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < H and 0 <= nj < W and B[ni, nj] and not visited[ni, nj]:
                            visited[ni, nj] = True
                            q.append((ni, nj))
                            coords.append((ni, nj))
                blobs.append(coords)

    blobs.sort(key=len, reverse=True)
    out = []
    for k in range(min(2, len(blobs))):
        M = np.zeros_like(B, dtype=bool)
        for (ii, jj) in blobs[k]:
            M[ii, jj] = True
        out.append(M)

    while len(out) < 2:  # pad if <2 blobs
        out.append(np.zeros_like(B, dtype=bool))
    return out[0], out[1]


def centroid(mask: np.ndarray):
    """
    Centroid (x,y) of a boolean mask in IMAGE coordinates (pixel centers).
    """
    ii, jj = np.where(mask)
    if ii.size == 0:
        return np.nan, np.nan
    x = jj.astype(float) + 0.5
    y = ii.astype(float) + 0.5
    return float(x.mean()), float(y.mean())


# --- Calibration ----------------------------------------------------------

def load_linear_calibration(csv_path: Path):
    """
    CSV with two columns: sep_px, diam_mm. Fit diam = alpha*sep + beta.
    Returns (alpha, beta).
    """
    data = np.genfromtxt(str(csv_path), delimiter=",", dtype=float, comments="#")
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    if data.shape[1] != 2:
        raise ValueError("Calibration CSV must have 2 columns: sep_px,diam_mm")
    sep = data[:, 0]
    dia = data[:, 1]
    A = np.vstack([sep, np.ones_like(sep)]).T
    alpha, beta = np.linalg.lstsq(A, dia, rcond=None)[0]
    return float(alpha), float(beta)


def pixels_to_mm(sep_px: float, mm_per_px: float = None, calib_ab: tuple = None):
    """
    Convert pixel separation to diameter in mm:
      - diam_mm = alpha*sep + beta  (if calib_ab provided)
      - diam_mm = sep * mm_per_px   (if mm_per_px provided)
    """
    if calib_ab is not None:
        a, b = calib_ab
        return a * sep_px + b
    if mm_per_px is not None:
        return sep_px * mm_per_px
    return float("nan")


# --- Panel splitting ------------------------------------------------------

def split_equal_panels(G: np.ndarray, n: int):
    """
    Split image into `n` equal-width vertical panels. Returns a list of sub-images.
    """
    H, W = G.shape
    w = W // n
    panels = []
    for k in range(n):
        x0 = k * w
        x1 = (k + 1) * w if k < n - 1 else W
        panels.append(G[:, x0:x1])
    return panels


# --- Core per panel -------------------------------------------------------

def process_panel(Gp: np.ndarray, args, panel_idx: int):
    """
    Full pipeline for a single panel:
      - threshold on inverted image (shadows are dark on bright screen)
      - keep two largest blobs (shadows)
      - measure separation and convert to mm
      - build a centered ROI
      - extract edges and fit ellipse (RANSAC + optional LM)
    Returns: (row_dict, plot_pack)
    """
    # Invert → shadows become bright for thresholding
    Ginv = 1.0 - Gp
    thr = args.threshold if args.threshold is not None else otsu_thresh01(Ginv)
    B0 = (Ginv >= thr).astype(float)

    # Clean binary (min_neighbors=1 keeps thin strokes; border cleared)
    B = preprocess_mask(B0, threshold=0.5, clear_border=True, min_neighbors=1)

    # Two largest blobs → shadows
    blob1, blob2 = largest_two_components(B > 0.5)
    x1, y1 = centroid(blob1)
    x2, y2 = centroid(blob2)
    if not np.isfinite(x1) or not np.isfinite(x2):
        raise RuntimeError("could not find two shadows; adjust threshold")

    # Separation (horizontal)
    sep_px = abs(x2 - x1)
    diam_mm = pixels_to_mm(sep_px, mm_per_px=args.mm_per_px, calib_ab=args._calib)

    # ROI centered between shadows, width ~ 0.9*sep, square ROI
    cxr = 0.5 * (x1 + x2)
    cyr = 0.5 * (y1 + y2)
    w_roi = max(20.0, 0.9 * sep_px)
    h_roi = w_roi
    H, W = Gp.shape
    x0 = max(0, int(cxr - w_roi / 2))
    x1b = min(W, int(cxr + w_roi / 2))
    y0 = max(0, int(cyr - h_roi / 2))
    y1b = min(H, int(cyr + h_roi / 2))

    ROI = Gp[y0:y1b, x0:x1b]
    ROIinv = 1.0 - ROI
    thr_roi = args.threshold if args.threshold is not None else otsu_thresh01(ROIinv)
    Broi0 = (ROIinv >= thr_roi).astype(float)
    # Preserve thin edges in ROI
    Broi = preprocess_mask(Broi0, threshold=0.5, clear_border=True, min_neighbors=0)

    # Edges in ROI (image coords), then shift to panel coords
    xr, yr = edge_points_ordered(Broi)
    if len(xr) < 20:
        xr, yr = edge_points(Broi, threshold=0.5)
    if len(xr) < 6:
        raise RuntimeError("not enough edge points in ROI for ellipse fit")

    x = xr + x0
    y = yr + y0

    # Robust fit + optional refinement
    cx, cy, a, b, th = fit_ellipse_ransac(
        x, y,
        iters=args.ransac_iters,
        sample_size=args.ransac_sample,
        inlier_thresh=args.ransac_thresh,
        min_inliers=max(20, len(x) // 3),
    )
    if args.refine:
        cx, cy, a, b, th = refine_ellipse_lm(x, y, cx, cy, a, b, th, iters=25, lam=1e-2)

    return {
        "panel": panel_idx,
        "sep_px": float(sep_px),
        "diam_mm": float(diam_mm),
        "cx": float(cx), "cy": float(cy),
        "a": float(a), "b": float(b), "theta_deg": float(np.degrees(th)),
        "roi": (int(x0), int(y0), int(x1b), int(y1b)),
        "centroids": (float(x1), float(y1), float(x2), float(y2)),
    }, (Gp, x, y, cx, cy, a, b, th, (x0, y0, x1b, y1b), (x1, y1, x2, y2))


# --- Overlay per panel ----------------------------------------------------

def save_panel_overlay(out_dir: Path, base: str, k: int, pack, args):
    """
    Save a PNG overlay for a processed panel.
    """
    Gp, x, y, cx, cy, a, b, th, roi, cents = pack
    (x0, y0, x1b, y1b) = roi
    (x1, y1, x2, y2) = cents
    Xf, Yf = ellipse_points_image(cx, cy, a, b, th, n=args.samples)

    fig, ax = plt.subplots(figsize=(4.6, 6.2), facecolor="white")
    ax.set_facecolor("white")
    ax.imshow(Gp, cmap="gray", origin="upper", vmin=0, vmax=1)

    # ROI
    ax.add_patch(plt.Rectangle((x0, y0), x1b - x0, y1b - y0,
                               fill=False, ec="gold", lw=1.2, ls="--"))

    # Shadows centroids + link
    ax.scatter([x1, x2], [y1, y2], s=36, c="cyan", label="shadows")
    ax.plot([x1, x2], [y1, y2], "c--", lw=1)

    # Edges + fitted ellipse
    ax.scatter(x, y, s=8, c="dodgerblue", label=f"edge points (n={len(x)})")
    ax.plot(Xf, Yf, "r", lw=2, label=f"fit θ={np.degrees(th):.1f}°")

    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base}_panel{k+1}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# --- CLI -----------------------------------------------------------------

def _build_argparser():
    p = argparse.ArgumentParser(
        prog="case_shadow_triptych",
        description="Measure two-shadow separation and fit ellipse on each panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", help="Input image (PNG/JPG) with 3 vertical panels.")
    p.add_argument("--panels", type=int, default=3, help="Number of vertical panels.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Foreground threshold on inverted gray; if omitted, Otsu is used.")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="Direct calibration factor (mm per pixel).")
    p.add_argument("--calib-csv", type=str, default="",
                   help="CSV with two columns: sep_px,diam_mm. Overrides --mm-per-px if present.")
    p.add_argument("--ransac-iters", type=int, default=1000)
    p.add_argument("--ransac-sample", type=int, default=8)
    p.add_argument("--ransac-thresh", type=float, default=0.20)
    p.add_argument("--refine", action="store_true", help="Run LM refinement after RANSAC.")
    p.add_argument("--samples", type=int, default=400, help="Ellipse sampling points for overlay.")
    p.add_argument("--save-dir", type=str, default="",
                   help="If set, save overlays here and a summary CSV.")
    return p


# --- Main ----------------------------------------------------------------

def main(argv=None):
    args = _build_argparser().parse_args(argv or sys.argv[1:])
    img_path = Path(args.image)
    G = imread_gray(img_path)
    base = img_path.stem

    # Calibration: either CSV or direct factor
    if args.calib_csv:
        args._calib = load_linear_calibration(Path(args.calib_csv))
    else:
        args._calib = None
        if args.mm_per_px is None:
            print("[warn] no calibration provided; diam_mm will be NaN")

    # Panels (equal width split)
    panels = split_equal_panels(G, args.panels)

    rows = []
    packs = []
    for k, Gp in enumerate(panels):
        try:
            row, pack = process_panel(Gp, args, k)
            rows.append(row)
            packs.append(pack)
            print(f"[ok] panel {k+1}: sep={row['sep_px']:.1f}px  "
                  f"diam≈{row['diam_mm']:.3f}mm  "
                  f"a={row['a']:.2f}, b={row['b']:.2f}, θ={row['theta_deg']:.1f}°")
        except Exception as e:
            print(f"[fail] panel {k+1}: {e}")

    # Save overlays and CSV if requested
    if args.save_dir:
        out_dir = Path(args.save_dir)
        for k, pack in enumerate(packs):
            _ = save_panel_overlay(out_dir, base, k, pack, args)

        header = ["panel", "sep_px", "diam_mm", "cx", "cy", "a", "b", "theta_deg"]
        out_csv = out_dir / f"{base}_summary.csv"
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in header) + "\n")
        print(f"[ok] overlays + summary -> {out_dir.resolve()}")

    return 0


# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
