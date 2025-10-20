"""
===========================================================
Practical Case — Two projected shadows per image (3 panels)
Ellipse-per-blob → per-image average A_k → final average
===========================================================

Pipeline (faithful to the physics):
  For each panel k:
    - detect the 2 darkest blobs anywhere in the panel (no fixed y-band);
    - fit ONE ellipse per blob with elli_fit (RANSAC, optional LM);
    - build A_k by averaging the 2 blobs in a canonical frame:
        * recenter each blob at (0,0),
        * convert sampled contours to polar (theta -> radius),
        * average radii bin-wise across the 2 blobs,
        * reconstruct A_k contour (x,y) around (0,0) and fit ellipse.
  Across the 3 panels:
    - average the three A_k the same polar way to get Ā,
    - fit the final ellipse on Ā (this is your final diameter).

Outputs:
  - panel overlays with the two blob fits
  - one overlay of the final averaged contour + ellipse
  - a CSV with per-blob params, per-panel A_k, and the final ellipse

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
    ellipse_points_image,  # (cx,cy,a,b,theta) -> sampled (X,Y)
)

# --- Basic I/O ------------------------------------------------------------

def imread_gray(path: Path) -> np.ndarray:
    """Read PNG/JPG -> grayscale float in [0,1] (matplotlib only)."""
    arr = plt.imread(str(path))
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        arr = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    arr = arr.astype(float)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr

def otsu_thresh01(g01: np.ndarray) -> float:
    """Otsu threshold on [0,1]."""
    g = np.clip(g01, 0, 1)
    hist, _ = np.histogram(g.ravel(), bins=256, range=(0,1))
    p = hist.astype(float)/max(1, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p*np.linspace(0,1,256))
    mu_t = mu[-1]
    sb2 = (mu_t*omega - mu)**2/(omega*(1-omega) + 1e-12)
    k = int(np.nanargmax(sb2))
    return (k+0.5)/256.0

# --- Illumination normalization (flat-field via local mean) ---------------

def _integral_image(A: np.ndarray) -> np.ndarray:
    return np.pad(A.cumsum(0).cumsum(1), ((1,0),(1,0)), mode="constant")

def _box_mean(A: np.ndarray, k: int) -> np.ndarray:
    H,W = A.shape
    II = _integral_image(A)
    k = max(3, int(k))
    y = np.arange(H); x = np.arange(W)
    Y0 = np.clip(y-k,0,H-1); Y1 = np.clip(y+k,0,H-1)
    X0 = np.clip(x-k,0,W-1); X1 = np.clip(x+k,0,W-1)
    Y0p, Y1p = Y0+1, Y1+1; X0p, X1p = X0+1, X1+1
    Y0g, X0g = np.meshgrid(Y0p, X0p, indexing="ij")
    Y1g, X1g = np.meshgrid(Y1p, X1p, indexing="ij")
    area = (Y1g-Y0g+1)*(X1g-X0g+1)
    S = II[Y1g, X1g]-II[Y0g-1, X1g]-II[Y1g, X0g-1]+II[Y0g-1, X0g-1]
    return S/area

def normalize_flatfield(G: np.ndarray, win_frac: float=0.12) -> np.ndarray:
    H,W = G.shape
    k = max(5, int(0.5*win_frac*min(H,W)))
    base = _box_mean(G, k)
    Gn = G/(base+1e-6)
    Gn -= Gn.min()
    m = Gn.max()
    if m > 0: Gn /= m
    return np.clip(Gn, 0, 1)

# --- Panel splitting ------------------------------------------------------

def split_equal_panels(G: np.ndarray, n: int):
    H,W = G.shape
    w = W//n
    panels=[]
    for k in range(n):
        x0 = k*w
        x1 = (k+1)*w if k < n-1 else W
        panels.append(G[:, x0:x1])
    return panels

# --- Blob detection (no fixed zone). Strategy: ----------------------------
# 1) Binarize inverted normalized image with Otsu (or user threshold).
# 2) Connected components (4-neigh), compute darkness & area.
# 3) Pick the two components with highest score:
#       score = sqrt(area) + α * mean_darkness
#    mean_darkness is computed on inverted gray (larger = darker).
# This is robust and unconstrained spatially.

def _connected_components_bool(B: np.ndarray):
    H,W = B.shape
    seen = np.zeros_like(B, bool)
    comps=[]
    for i in range(H):
        for j in range(W):
            if B[i,j] and not seen[i,j]:
                q=[(i,j)]; seen[i,j]=True; coords=[(i,j)]
                while q:
                    ci,cj = q.pop()
                    for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
                        ni,nj=ci+di,cj+dj
                        if 0<=ni<H and 0<=nj<W and B[ni,nj] and not seen[ni,nj]:
                            seen[ni,nj]=True; q.append((ni,nj)); coords.append((ni,nj))
                M = np.zeros_like(B,bool)
                for (ii,jj) in coords: M[ii,jj]=True
                comps.append(M)
    return comps

def _blob_features(M: np.ndarray, Ginv: np.ndarray):
    ii,jj = np.where(M)
    if ii.size==0: return None
    area = float(ii.size)
    x = jj.astype(float)+0.5; y = ii.astype(float)+0.5
    cx,cy = float(x.mean()), float(y.mean())
    mean_dark = float(Ginv[ii,jj].mean())
    return dict(mask=M, area=area, cx=cx, cy=cy, mean_dark=mean_dark)

def detect_two_blobs_anywhere(
    Gp: np.ndarray,
    *,
    thr: float = None,
    alpha_dark: float = 2.0,
    min_area: int = 40,
    thr_widen: float = 0.18,
    # NEW robust filters (tune if needed)
    border_margin: int = 6,       # pixels near image edges to reject
    max_bbox_frac: float = 0.75,  # if bbox spans > this fraction of H or W → reject
    min_roundness: float = 0.42,  # 4πA/P^2 threshold; raise if still catching strips
    debug_ax=None
):
    """
    Robustly find TWO blob masks anywhere in the panel (no positional priors),
    while rejecting frame-like components that touch borders or are very elongated.
    """

    # --- helpers ----------------------------------------------------------
    def _touches_border(M: np.ndarray, m: int) -> bool:
        H, W = M.shape
        if m <= 0: 
            return False
        return (
            M[:m, :].any() or M[-m:, :].any() or
            M[:, :m].any() or M[:, -m:].any()
        )

    def _perimeter4(M: np.ndarray) -> int:
        up    = np.zeros_like(M); up[1:,  :]  = M[:-1, :]
        down  = np.zeros_like(M); down[:-1,:] = M[1:,  :]
        left  = np.zeros_like(M); left[:, 1:] = M[:, :-1]
        right = np.zeros_like(M); right[:, :-1]= M[:, 1:]
        interior = up & down & left & right
        edge = M & (~interior)
        return int(edge.sum())

    def _roundness(M: np.ndarray) -> float:
        A = float(M.sum())
        if A <= 0: 
            return 0.0
        P = float(_perimeter4(M))
        if P <= 0:
            return 0.0
        return float(4.0 * np.pi * A / (P * P))

    # --- (1) normalize + invert -----------------------------------------
    Gn = normalize_flatfield(Gp, 0.12)
    Ginv = 1.0 - Gn
    base_thr = float(otsu_thresh01(Ginv)) if thr is None else float(thr)

    # threshold sweep grids
    H, W = Gp.shape
    deltas = [0.0, +0.5*thr_widen, -0.5*thr_widen, +1.0*thr_widen, -1.0*thr_widen]
    min_neighbors_grid = [2, 1, 0]
    clear_border_grid = [True, False]
    min_area_grid = [min_area, max(8, int(0.5*min_area)), max(6, int(0.25*min_area))]

    def _features_after_filters(Bbin, min_area_try):
        comps = _connected_components_bool(Bbin)
        feats = []
        for M in comps:
            A = float(M.sum())
            if A < min_area_try:
                continue
            # reject if touches border
            if _touches_border(M, border_margin):
                continue
            # reject if bbox spans too much (typical of frame)
            ii, jj = np.where(M)
            y0, y1 = int(ii.min()), int(ii.max())
            x0, x1 = int(jj.min()), int(jj.max())
            h = (y1 - y0 + 1) / max(1, H)
            w = (x1 - x0 + 1) / max(1, W)
            if h > max_bbox_frac or w > max_bbox_frac:
                continue
            # reject if not round enough
            if _roundness(M) < min_roundness:
                continue

            # score remaining candidates
            cx = float(jj.mean() + 0.5)
            cy = float(ii.mean() + 0.5)
            md = float(Ginv[ii, jj].mean())  # darker → larger
            score = np.sqrt(A) + alpha_dark * md
            feats.append(dict(mask=M, area=A, cx=cx, cy=cy, dark=md, score=float(score)))
        feats.sort(key=lambda d: d["score"], reverse=True)
        return feats

    # --- (2) sweep configs until ≥ 2 components --------------------------
    best = None
    for dthr in deltas:
        t = np.clip(base_thr + dthr, 0.0, 1.0)
        B0 = (Ginv >= t).astype(float)
        for mn in min_neighbors_grid:
            for cb in clear_border_grid:
                B = preprocess_mask(B0, threshold=0.5, clear_border=cb, min_neighbors=mn)
                for min_area_try in min_area_grid:
                    feats = _features_after_filters(B > 0.5, min_area_try)
                    if len(feats) >= 2:
                        cand = feats[:2]
                        pair_score = cand[0]["score"] + cand[1]["score"]
                        if (best is None) or (pair_score > best[0]):
                            best = (pair_score, cand, dict(thr=t, mn=mn, cb=cb, min_area=min_area_try))
        if best is not None:
            break  # early exit once we have a good pair

    if best is not None:
        _, cand, _used = best
        f1, f2 = cand[0], cand[1]
        if debug_ax is not None:
            for f in cand:
                ii, jj = np.where(f["mask"])
                x0, x1 = int(jj.min()), int(jj.max())
                y0, y1 = int(ii.min()), int(ii.max())
                debug_ax.add_patch(plt.Rectangle((x0, y0), x1-x0+1, y1-y0+1,
                                                 fill=False, ec="yellow", lw=1.2, ls="--"))
        return f1["mask"], f2["mask"], (f1["cx"], f1["cy"], f2["cx"], f2["cy"])

    # --- (3) fallback: two darkest local minima (kept from before) -------
    k = max(5, int(0.04 * min(H, W)))
    sm = _box_mean(Ginv, k)
    flat = sm.ravel()
    idx2 = np.argpartition(-flat, 2)[:2]
    idx2 = idx2[np.argsort(-flat[idx2])]
    masks = []; centers = []
    r = max(6, int(0.06 * min(H, W)))
    YY, XX = np.mgrid[0:H, 0:W]
    for idx in idx2:
        cy, cx = np.unravel_index(int(idx), sm.shape)
        # also respect border margin in fallback
        if (cy < border_margin or cy >= H-border_margin or
            cx < border_margin or cx >= W-border_margin):
            continue
        M = ((XX - cx)**2 + (YY - cy)**2 <= r*r)
        masks.append(M); centers.append((float(cx), float(cy)))
    if len(masks) == 2:
        if debug_ax is not None:
            for (cx, cy) in centers:
                debug_ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="lime", lw=1.2))
        return masks[0], masks[1], (centers[0][0], centers[0][1], centers[1][0], centers[1][1])

    raise RuntimeError("blob detection: <2 components (even after fallback)")

# --- Per-blob ellipse fit -------------------------------------------------

def fit_blob_ellipse(Gp: np.ndarray, Mblob: np.ndarray, *, ransac_iters=1200,
                     ransac_sample=8, ransac_thresh=0.20, refine=True, samples=400):
    """
    Crop around the blob, take edges, fit ellipse (RANSAC + optional LM).
    Returns (cx,cy,a,b,theta), abs coords in panel, and sampled points.
    """
    ii,jj = np.where(Mblob)
    x0,x1 = int(jj.min()), int(jj.max())+1
    y0,y1 = int(ii.min()), int(ii.max())+1
    crop = Mblob[y0:y1, x0:x1].astype(float)
    # Extract contour pixels of the blob mask
    xr, yr = edge_points_ordered(crop)
    if len(xr) < 8:
        xr, yr = edge_points(crop, threshold=0.5)
    if len(xr) < 6:
        raise RuntimeError("blob edge extraction failed")
    x = xr + x0; y = yr + y0
    cx, cy, a, b, th = fit_ellipse_ransac(
        x, y, iters=ransac_iters, sample_size=ransac_sample,
        inlier_thresh=ransac_thresh, min_inliers=max(20, len(x)//3),
    )
    if refine:
        cx,cy,a,b,th = refine_ellipse_lm(x,y,cx,cy,a,b,th,iters=25,lam=1e-2)
    Xs, Ys = ellipse_points_image(cx,cy,a,b,th,n=samples)
    return (cx,cy,a,b,th), (Xs,Ys), (x,y)

# --- Polar averaging helpers ---------------------------------------------

def polar_profile(X: np.ndarray, Y: np.ndarray, nbins=360):
    """Return (theta_bins, r_mean, counts) for points around (0,0)."""
    ang = np.arctan2(Y, X)
    rad = np.hypot(X, Y)
    tb = np.linspace(-np.pi, np.pi, nbins, endpoint=False)
    idx = ((ang - tb[0])/(2*np.pi) * nbins).astype(int) % nbins
    rsum = np.zeros(nbins); n = np.zeros(nbins, int)
    np.add.at(rsum, idx, rad)
    np.add.at(n,    idx, 1)
    rmean = np.divide(rsum, np.maximum(n,1), out=np.zeros_like(rsum), where=n>0)
    return tb, rmean, n

def average_two_blobs_to_Ak(ell1_pts: tuple, ell2_pts: tuple, nbins=360):
    """
    Build A_k by averaging radii of the two blobs in a canonical centered frame.
    Each blob contour is first centered at (0,0) (remove its centroid).
    """
    (X1,Y1),(X2,Y2) = ell1_pts, ell2_pts
    # recenter each blob at its own centroid to get a canonical shape
    c1x,c1y = float(np.mean(X1)), float(np.mean(Y1))
    c2x,c2y = float(np.mean(X2)), float(np.mean(Y2))
    X1c, Y1c = (X1-c1x), (Y1-c1y)
    X2c, Y2c = (X2-c2x), (Y2-c2y)

    tb1, r1, n1 = polar_profile(X1c, Y1c, nbins=nbins)
    tb2, r2, n2 = polar_profile(X2c, Y2c, nbins=nbins)
    # bins have same tb by construction
    with np.errstate(invalid="ignore"):
        r_mean = np.nanmean(
            np.vstack([np.where(n1>0, r1, np.nan),
                       np.where(n2>0, r2, np.nan)]), axis=0)
    mask = ~np.isnan(r_mean)
    Xk = r_mean[mask]*np.cos(tb1[mask])
    Yk = r_mean[mask]*np.sin(tb1[mask])
    return Xk, Yk  # centered A_k points

def average_Aks(Ak_list, nbins=360):
    """Average several A_k (already centered) in polar bins."""
    # build a global polar table
    all_tb = np.linspace(-np.pi, np.pi, nbins, endpoint=False)
    r_acc = []; m_acc = []
    for Xk, Yk in Ak_list:
        tb, r, n = polar_profile(Xk, Yk, nbins=nbins)
        r_acc.append(np.where(n>0, r, np.nan))
    R = np.vstack(r_acc)
    with np.errstate(invalid="ignore"):
        r_mean = np.nanmean(R, axis=0)
    mask = ~np.isnan(r_mean)
    X = r_mean[mask]*np.cos(all_tb[mask])
    Y = r_mean[mask]*np.sin(all_tb[mask])
    return X, Y

# --- Panel processing -----------------------------------------------------

def process_panel(Gp: np.ndarray, args, k: int):
    """
    For panel k:
      - detect two blobs anywhere;
      - fit 1 ellipse per blob;
      - build A_k from the two ellipses (centered, polar-averaged).
    """
    # BEFORE fitting blobs
    dbg_ax = (plt.gca() if getattr(args, "debug_detector", False) else None)
    M1, M2, (x1,y1,x2,y2) = detect_two_blobs_anywhere(
    Gp,
    thr=(args.threshold if args.threshold is not None else None),
    alpha_dark=args.alpha_dark,
    min_area=args.min_area,
    thr_widen=args.thr_widen,
    debug_ax=dbg_ax,
)

    # fit each blob
    (cx1,cy1,a1,b1,th1),(E1x,E1y),(e1x,e1y) = fit_blob_ellipse(
        Gp, M1, ransac_iters=args.ransac_iters, ransac_sample=args.ransac_sample,
        ransac_thresh=args.ransac_thresh, refine=args.refine, samples=args.samples
    )
    (cx2,cy2,a2,b2,th2),(E2x,E2y),(e2x,e2y) = fit_blob_ellipse(
        Gp, M2, ransac_iters=args.ransac_iters, ransac_sample=args.ransac_sample,
        ransac_thresh=args.ransac_thresh, refine=args.refine, samples=args.samples
    )

    # per-image averaged contour A_k in canonical frame
    AkX, AkY = average_two_blobs_to_Ak((E1x,E1y),(E2x,E2y), nbins=args.avg_bins)

   # fit ellipse on A_k (centered)
    cxk, cyk, ak, bk, thk = fit_ellipse_ransac(
        AkX, AkY,
        iters=600,
        sample_size=8,
        inlier_thresh=0.12,
        min_inliers=max(20, AkX.size // 3)
    )
    if args.refine:
        cxk, cyk, ak, bk, thk = refine_ellipse_lm(
            AkX, AkY, cxk, cyk, ak, bk, thk,
            iters=25, lam=1e-2
        )


    row = {
        "panel": k+1,
        # blob 1
        "b1_cx": cx1, "b1_cy": cy1, "b1_a": a1, "b1_b": b1, "b1_theta_deg": np.degrees(th1),
        # blob 2
        "b2_cx": cx2, "b2_cy": cy2, "b2_a": a2, "b2_b": b2, "b2_theta_deg": np.degrees(th2),
        # per-image average Ak (centered)
        "Ak_a": ak, "Ak_b": bk, "Ak_theta_deg": np.degrees(thk),
        # separation (can be useful for diameter with external calibration)
        "sep_px": abs(x2-x1),
    }

    pack = dict(
        Gp=Gp, blobs=[(M1,(cx1,cy1,a1,b1,th1),(E1x,E1y),(e1x,e1y)),
                      (M2,(cx2,cy2,a2,b2,th2),(E2x,E2y),(e2x,e2y))],
        Ak=(AkX,AkY, (cxk,cyk,ak,bk,thk))
    )
    return row, pack

# --- Overlays -------------------------------------------------------------

def draw_panel_overlay(out_dir: Path, base: str, k: int, pack, args):
    Gp = pack["Gp"]; (AkX,AkY,(cxk,cyk,ak,bk,thk)) = pack["Ak"]
    fig, ax = plt.subplots(figsize=(4.8, 8.0), facecolor="white")
    ax.imshow(Gp, cmap="gray", origin="upper", vmin=0, vmax=1)
    # blobs
    for (Mb,(cx,cy,a,b,th),(Ex,Ey),(ex,ey)) in pack["blobs"]:
        ax.scatter(ex,ey, s=6, c="dodgerblue", alpha=0.7, label="blob edges" if k==0 else None)
        Xf,Yf = ellipse_points_image(cx,cy,a,b,th,n=args.samples)
        ax.plot(Xf,Yf,"r",lw=2, alpha=0.9, label="ellipse fit" if k==0 else None)
    ax.set_aspect("equal","box")
    ax.axis("off")
    if k==0:
        ax.legend(frameon=False, loc="upper left")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base}_panel{k+1}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

def draw_global_overlay(out_dir: Path, base: str, Ak_list, args, final_params):
    fig, ax = plt.subplots(figsize=(6.0,6.0), facecolor="white")
    for (AkX,AkY) in Ak_list:
        ax.plot(AkX,AkY, ".", ms=2, alpha=0.5, label="A_k contour" if ax.lines==[] else None)
    (cgx,cgy,ag,bg,thg) = final_params
    Xf,Yf = ellipse_points_image(cgx,cgy,ag,bg,thg,n=600)
    ax.plot(Xf,Yf,"r",lw=2, label="final avg ellipse")
    ax.axhline(0,color="k",alpha=0.2,ls=":")
    ax.axvline(0,color="k",alpha=0.2,ls=":")
    ax.set_aspect("equal","box")
    ax.set_title("Final averaged contour Ā (canonical frame)")
    ax.legend(frameon=False, loc="best")
    out = Path(out_dir) / f"{base}_Abar.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

# --- Calibration helpers (optional) ---------------------------------------

def load_linear_calibration(csv_path: Path):
    data = np.genfromtxt(str(csv_path), delimiter=",", dtype=float, comments="#")
    if data.ndim==1: data=data.reshape(-1,2)
    if data.shape[1]!=2: raise ValueError("CSV must be: sep_px,diam_mm")
    sep = data[:,0]; dia = data[:,1]
    A = np.vstack([sep, np.ones_like(sep)]).T
    a,b = np.linalg.lstsq(A, dia, rcond=None)[0]
    return float(a), float(b)

def px_to_mm(sep_px: float, mm_per_px: float=None, calib=None):
    if calib is not None:
        a,b = calib; return a*sep_px + b
    if mm_per_px is not None:
        return sep_px * mm_per_px
    return float("nan")

# --- CLI ------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        prog="case_two_shadow_average",
        description="Fit one ellipse per blob, make per-image average Ak, then final averaged ellipse across 3 images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", help="Grayscale triptych (3 vertical panels).")
    p.add_argument("--panels", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None, help="Foreground threshold on inverted gray; else Otsu.")
    p.add_argument("--mm-per-px", type=float, default=None, help="Optional: convert sep_px to mm.")
    p.add_argument("--calib-csv", type=str, default="", help="Optional CSV sep_px,diam_mm.")
    p.add_argument("--ransac-iters", type=int, default=1200)
    p.add_argument("--ransac-sample", type=int, default=8)
    p.add_argument("--ransac-thresh", type=float, default=0.20)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--samples", type=int, default=400, help="Ellipse sampling for overlays/averaging.")
    p.add_argument("--avg-bins", type=int, default=360, help="Angle bins for polar averaging.")
    p.add_argument("--save-dir", type=str, default="out_case")
    p.add_argument("--min-area", type=int, default=40,
               help="Minimal area (px) for blob candidates before fallback.")
    p.add_argument("--alpha-dark", type=float, default=2.0,
               help="Weight of darkness in blob score (higher → prefers darker).")
    p.add_argument("--thr-widen", type=float, default=0.18,
               help="Threshold sweep half-span around Otsu (0..1). Larger → more relaxed.")
    p.add_argument("--debug-detector", action="store_true",
               help="Draw rectangles/minima used by detector.")

    return p

# --- Main -----------------------------------------------------------------

def main(argv=None):
    args = _build_parser().parse_args(argv or sys.argv[1:])
    img_path = Path(args.image)
    G = imread_gray(img_path)
    base = img_path.stem

    # set up calibration if provided
    calib = None
    if args.calib_csv:
        calib = load_linear_calibration(Path(args.calib_csv))

    # split panels
    panels = split_equal_panels(G, args.panels)

    rows = []
    packs = []
    Ak_list = []

    for k, Gp in enumerate(panels):
        try:
            row, pack = process_panel(Gp, args, k)
            # enrich with diameter if calibration
            row["diam_mm"] = px_to_mm(row["sep_px"], mm_per_px=args.mm_per_px, calib=calib)
            rows.append(row); packs.append(pack)
            # collect Ak
            Ak_list.append((pack["Ak"][0], pack["Ak"][1]))
            print(f"[ok] panel {k+1}: "
                  f"b1(a={row['b1_a']:.2f}, b={row['b1_b']:.2f}), "
                  f"b2(a={row['b2_a']:.2f}, b={row['b2_b']:.2f})  "
                  f"| Ak(a={row['Ak_a']:.2f}, b={row['Ak_b']:.2f})  "
                  f"| sep={row['sep_px']:.1f}px  diam≈{row['diam_mm']:.3f}mm")
        except Exception as e:
            print(f"[fail] panel {k+1}: {e}")

    # save per-panel overlays
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for k, pack in enumerate(packs):
        draw_panel_overlay(out_dir, base, k, pack, args)

    # final average across A_k
    if Ak_list:
        Xbar, Ybar = average_Aks(Ak_list, nbins=args.avg_bins)
        # fit final ellipse in canonical frame (center≈0)
        cgx,cgy,ag,bg,thg = fit_ellipse_ransac(
            Xbar, Ybar, iters=800, sample_size=8, inlier_thresh=0.12,
            min_inliers=max(20, Xbar.size//3)
        )
        if args.refine:
            cgx,cgy,ag,bg,thg = refine_ellipse_lm(Xbar,Ybar,cgx,cgy,ag,bg,thg,iters=25,lam=1e-2)

        print(f"[GLOBAL] Ā ellipse: ā={ag:.2f}, b̄={bg:.2f}, θ̄={np.degrees(thg):.1f}°  "
              f"(major diameter ≈ {2*ag:.2f} px, minor ≈ {2*bg:.2f} px)")
        draw_global_overlay(out_dir, base, Ak_list, args, (cgx,cgy,ag,bg,thg))

        # CSV
        csv = out_dir / f"{base}_summary.csv"
        with csv.open("w", encoding="utf-8") as f:
            # per-blob + Ak per panel
            cols = ["panel",
                    "b1_a","b1_b","b1_theta_deg",
                    "b2_a","b2_b","b2_theta_deg",
                    "Ak_a","Ak_b","Ak_theta_deg",
                    "sep_px","diam_mm"]
            f.write(",".join(cols)+"\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols)+"\n")
            # final line
            f.write("# final averaged ellipse in canonical frame\n")
            f.write("# a_bar,b_bar,theta_deg,major_diam_px,minor_diam_px\n")
            f.write(f"{ag:.6f},{bg:.6f},{np.degrees(thg):.6f},{2*ag:.6f},{2*bg:.6f}\n")
        print(f"[ok] overlays + summary -> {out_dir.resolve()}")

    return 0

# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
