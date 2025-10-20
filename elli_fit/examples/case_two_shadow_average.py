#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
Two-Shadow Averaging + Ellipse Fit (robust, NumPy-only)
===========================================================

But
---
Pour chaque panneau vertical d'une image triptyque :
  1) détecter automatiquement *deux* ombres (pics LoG, robustes au bord),
  2) extraire les bords de chaque blob, fitter une ellipse par blob,
  3) calculer A_k = moyenne des diamètres équivalents des 2 blobs,
  4) après les 3 panneaux, renvoyer la moyenne des A_k,
  5) sauvegarder overlay(s) et un CSV récapitulatif.

Dépendances
-----------
- numpy, matplotlib
- ta librairie `elli_fit.core` (déjà installée dans ton projet) :
    preprocess_mask, edge_points, fit_ellipse_ransac,
    refine_ellipse_lm, ellipse_points_image

Usage
-----
python3 case_two_shadow_average.py IMG.png \
  --panels 3 --save-dir out_case --refine \
  --mm-per-px 0.045 --debug

Notes
-----
- La détection ne contraint *pas* la zone : on cherche *partout*.
- La routine renvoie *toujours* 2 masques (fallback → disques).
- Diamètre “équivalent” d’un blob = 2 * sqrt(a*b) (aire conservée).
"""

# --- Imports --------------------------------------------------------------

import sys
import os
import math
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from elli_fit.core import (
    preprocess_mask,
    edge_points,
    fit_ellipse_ransac,
    refine_ellipse_lm,
    ellipse_points_image,
)

# --- Petites briques d’imagerie (NumPy-only) ------------------------------

def imread_gray(path: Path) -> np.ndarray:
    """Lire PNG/JPG en niveau de gris [0,1] via matplotlib (sans dépendances)."""
    arr = plt.imread(str(path))
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    arr = arr.astype(float)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _box_mean(A: np.ndarray, r: int) -> np.ndarray:
    """Moyenne locale (fenêtre carrée) par image intégrale (rapide, NumPy)."""
    r = int(max(1, r))
    H, W = A.shape
    II = np.pad(A.cumsum(0).cumsum(1), ((1,0),(1,0)), mode="constant")
    y0 = np.clip(np.arange(H) - r, 0, H-1)
    y1 = np.clip(np.arange(H) + r, 0, H-1)
    x0 = np.clip(np.arange(W) - r, 0, W-1)
    x1 = np.clip(np.arange(W) + r, 0, W-1)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    Y0 += 1; X0 += 1; Y1 += 1; X1 += 1
    area = (Y1 - Y0 + 1) * (X1 - X0 + 1)
    S = II[Y1, X1] - II[Y0-1, X1] - II[Y1, X0-1] + II[Y0-1, X0-1]
    return S / area


def normalize_flatfield(G: np.ndarray, win_frac: float = 0.12) -> np.ndarray:
    """Normalisation illumination (divide-by-local-mean) + rescale [0,1]."""
    H, W = G.shape
    r = max(5, int(round(win_frac * min(H, W))))
    base = _box_mean(G, r)
    Gn = G / (base + 1e-6)
    Gn -= Gn.min()
    m = Gn.max()
    if m > 0:
        Gn /= m
    return np.clip(Gn, 0.0, 1.0)


def gaussian_blur_separable(G: np.ndarray, sigma: float) -> np.ndarray:
    """Flou gaussien simple (séparable, 1D conv), suffisamment rapide en 2D."""
    sigma = float(max(0.6, sigma))
    R = int(max(3, round(3*sigma)))
    x = np.arange(-R, R+1, dtype=float)
    k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
    A = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, G)
    B = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, A)
    return B


def log_response(G: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """Approx. Laplacien-of-Gaussian par Difference-of-Gaussians (DoG)."""
    B1 = gaussian_blur_separable(G, sigma1)
    B2 = gaussian_blur_separable(G, sigma2)
    # Pour nos ombres (sombres → brillantes après inversion), on veut les *pics* :
    # travailler sur Ginv et prendre DoG>0.
    return B1 - B2  # sigma1 < sigma2 → pics positifs aux structures sombres


# --- Détection robuste de 2 ombres (jamais vide) -------------------------

def _nms_greedy_peaks(A: np.ndarray, k: int, suppr_r: int):
    """
    Greedy non-maximum suppression : choisir k maxima, en annulant un disque
    de rayon suppr_r autour de chaque maximum successif.
    """
    A = A.copy()
    H, W = A.shape
    peaks = []
    YY, XX = np.mgrid[0:H, 0:W]
    for _ in range(k):
        idx = int(np.argmax(A))
        v = float(A.ravel()[idx])
        if not np.isfinite(v) or v <= 0:
            break
        y, x = np.unravel_index(idx, A.shape)
        peaks.append((x, y, v))
        mask = (XX - x)**2 + (YY - y)**2 <= suppr_r**2
        A[mask] = -np.inf
    return peaks


def _region_grow_threshold(G, cx, cy, thr, rmax):
    """
    Croissance de région (BFS) autour (cx,cy) sur G (float), en gardant les
    pixels >= thr et dans un disque de rayon rmax. Renvoie masque booléen.
    """
    H, W = G.shape
    cx, cy = int(round(cx)), int(round(cy))
    if not (0 <= cx < W and 0 <= cy < H):
        return np.zeros_like(G, bool)
    M = np.zeros((H, W), dtype=bool)
    q = [(cy, cx)]
    M[cy, cx] = True
    r2 = rmax * rmax
    while q:
        iy, ix = q.pop()
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < H and 0 <= nx < W and (not M[ny, nx]):
                if (nx - cx)**2 + (ny - cy)**2 <= r2 and G[ny, nx] >= thr:
                    M[ny, nx] = True
                    q.append((ny, nx))
    return M


def detect_two_blobs_anywhere(
    G: np.ndarray,
    *,
    pad: int = 32,
    sigma_small: float = 2.0,
    sigma_large: float = 6.0,
    k_peaks: int = 6,
    nms_radius: int = 20,
    grow_rel: float = 0.55,
    rmax_rel: float = 0.10,
    min_area: int = 30,
    fallback_r_rel: float = 0.06,
    debug_ax=None,
):
    """
    Renvoie 2 masques (toujours), détectés via pics LoG + croissance de région.
    - pad miroir large → supporte blobs bord image
    - si croissance échoue → disques fallback
    """
    # 1) normaliser + inverser + pad
    Gp = np.pad(G, ((pad,pad),(pad,pad)), mode="reflect")
    Gn = normalize_flatfield(Gp, 0.12)
    Ginv = 1.0 - Gn

    # 2) LoG approx (DoG) + rectif signes
    DoG = log_response(Ginv, sigma_small, sigma_large)
    DoG[DoG < 0] = 0.0

    # 3) pics globaux (NMS)
    peaks = _nms_greedy_peaks(DoG, k=k_peaks, suppr_r=nms_radius)
    if len(peaks) == 0:
        # fallback brutal : pics sur Ginv lissé
        Sm = _box_mean(Ginv, max(5, int(0.03*min(Ginv.shape))))
        peaks = _nms_greedy_peaks(Sm, k=2, suppr_r=max(10, int(0.05*min(Ginv.shape))))

    # 4) construire 2 blobs “les plus séparés horizontalement”
    H, W = Gp.shape
    rmax = int(max(12, rmax_rel * min(H, W)))
    cand = []
    for (x, y, v) in peaks:
        thr = grow_rel * v
        M = _region_grow_threshold(DoG, x, y, thr, rmax)
        if M.sum() < min_area:
            # élargir ou fallback disque
            rfb = int(max(10, fallback_r_rel * min(H, W)))
            YY, XX = np.mgrid[0:H, 0:W]
            M = ((XX - x)**2 + (YY - y)**2 <= rfb*rfb)
        ii, jj = np.where(M)
        if ii.size == 0:
            continue
        cx = float(jj.mean()); cy = float(ii.mean())
        cand.append(dict(mask=M, cx=cx, cy=cy))

    if len(cand) == 0:
        # force deux disques au centre
        h, w = Gp.shape
        d1 = np.zeros_like(Gp, bool); d2 = np.zeros_like(Gp, bool)
        YY, XX = np.mgrid[0:h, 0:w]
        rfb = int(max(10, fallback_r_rel * min(h, w)))
        d1[(XX - int(w*0.33))**2 + (YY - int(h*0.5))**2 <= rfb*rfb] = True
        d2[(XX - int(w*0.66))**2 + (YY - int(h*0.5))**2 <= rfb*rfb] = True
        cand = [dict(mask=d1, cx=w*0.33, cy=h*0.5), dict(mask=d2, cx=w*0.66, cy=h*0.5)]

    # 5) choisir la paire avec plus grande séparation horizontale
    best = None
    for i in range(len(cand)):
        for j in range(i+1, len(cand)):
            dx = abs(cand[i]["cx"] - cand[j]["cx"])
            if (best is None) or (dx > best[0]):
                best = (dx, cand[i], cand[j])
    _, f1, f2 = best

    # 6) retirer le padding
    Hp, Wp = Gp.shape
    H0, W0 = G.shape
    out = []
    for f in (f1, f2):
        Mpad = f["mask"]
        M = Mpad[pad:pad+H0, pad:pad+W0]
        cx = f["cx"] - pad
        cy = f["cy"] - pad
        out.append((M, float(cx), float(cy)))

        if debug_ax is not None:
            ii, jj = np.where(M)
            if ii.size:
                x0, x1 = int(jj.min()), int(jj.max())
                y0, y1 = int(ii.min()), int(ii.max())
                debug_ax.add_patch(
                    plt.Rectangle((x0,y0), x1-x0+1, y1-y0+1, fill=False, ec="deepskyblue", lw=2)
                )

    return out[0], out[1]


# --- Mesures / pipeline panneau ------------------------------------------

def equivalent_diameter(a: float, b: float) -> float:
    """Diamètre équivalent aire: 2*sqrt(a*b)."""
    return 2.0 * math.sqrt(max(a, 1e-9) * max(b, 1e-9))


def fit_blob_ellipse(G: np.ndarray, M: np.ndarray, refine: bool):
    """Bords → fit ellipse robuste (+ LM optionnel)."""
    # bords du masque (dans coords image)
    edges = preprocess_mask(M.astype(float), threshold=0.5, clear_border=False, min_neighbors=0)
    x, y = edge_points(edges, threshold=0.5)
    if len(x) < 6:
        # élargir par un léger dilate numérique : anneau de 1 px
        Y, X = np.where(M)
        if Y.size:
            x = X.astype(float) + 0.5
            y = Y.astype(float) + 0.5
    if len(x) < 6:
        raise RuntimeError("Blob edges too few")

    cx, cy, a, b, th = fit_ellipse_ransac(
        x, y, iters=800, sample_size=8, inlier_thresh=0.15, min_inliers=max(20, len(x)//3)
    )
    if refine:
        cx, cy, a, b, th = refine_ellipse_lm(x, y, cx, cy, a, b, th, iters=20, lam=1e-2)
    return (cx, cy, a, b, th), (x, y)


def split_equal_panels(G: np.ndarray, n: int):
    """Découper l’image en n panneaux verticaux égaux."""
    H, W = G.shape
    w = W // n
    out = []
    for k in range(n):
        x0 = k*w
        x1 = (k+1)*w if k < n-1 else W
        out.append(G[:, x0:x1])
    return out


# --- Overlay helpers ------------------------------------------------------

def draw_blob_fit(ax, G, M, fit_params, label_prefix="blob"):
    (cx, cy, a, b, th) = fit_params
    Xf, Yf = ellipse_points_image(cx, cy, a, b, th, n=400)
    ax.imshow(G, cmap="gray", origin="upper", vmin=0, vmax=1)
    # bord du masque (debug)
    ii, jj = np.where(M)
    ax.scatter(jj+0.5, ii+0.5, s=6, c="dodgerblue", label=f"{label_prefix} edges")
    ax.plot(Xf, Yf, "r", lw=2, label=f"{label_prefix} ellipse fit")
    ax.set_aspect("equal", "box")
    ax.axis("off")


# --- CLI -----------------------------------------------------------------

def _build_argparser():
    p = argparse.ArgumentParser(
        prog="case_two_shadow_average",
        description="Detect 2 shadows per panel, fit ellipse per blob, average per panel and overall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", help="Input grayscale image (triptych).")
    p.add_argument("--panels", type=int, default=3)
    p.add_argument("--save-dir", type=str, default="out_case")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="Optional scale: diameter_mm = pixels * mm_per_px (reported additionally).")
    p.add_argument("--refine", action="store_true", help="LM refinement after RANSAC.")
    # detection params
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--sigma-small", type=float, default=2.0)
    p.add_argument("--sigma-large", type=float, default=6.0)
    p.add_argument("--grow-rel", type=float, default=0.55)
    p.add_argument("--rmax-rel", type=float, default=0.10)
    p.add_argument("--min-area", type=int, default=30)
    p.add_argument("--debug", action="store_true")
    return p


# --- Main routine --------------------------------------------------------

def main(argv=None):
    args = _build_argparser().parse_args(argv or sys.argv[1:])

    img_path = Path(args.image)
    G = imread_gray(img_path)
    panels = split_equal_panels(G, args.panels)
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    Ak_list = []

    for k, Gp in enumerate(panels):
        # Détection 2 blobs (jamais vide)
        (M1, cx1, cy1), (M2, cx2, cy2) = detect_two_blobs_anywhere(
            Gp,
            pad=args.pad,
            sigma_small=args.sigma_small,
            sigma_large=args.sigma_large,
            grow_rel=args.grow_rel,
            rmax_rel=args.rmax_rel,
            min_area=args.min_area,
            debug_ax=None,
        )

        # Fit par blob
        (fit1, (x1e, y1e)) = fit_blob_ellipse(Gp, M1, refine=args.refine)
        (fit2, (x2e, y2e)) = fit_blob_ellipse(Gp, M2, refine=args.refine)

        d1 = equivalent_diameter(fit1[2], fit1[3])  # 2*sqrt(a*b)
        d2 = equivalent_diameter(fit2[2], fit2[3])
        Ak = 0.5 * (d1 + d2)
        Ak_list.append(Ak)

        # Overlay panneau
        fig, ax = plt.subplots(figsize=(3.0, 9.0), facecolor="white")
        ax.imshow(Gp, cmap="gray", origin="upper", vmin=0, vmax=1)
        # centroids approx
        ax.scatter([cx1, cx2], [cy1, cy2], s=40, c="cyan", zorder=3, label="shadows (centroids)")
        # bords & ellipses
        for (M, fitp, lbl) in [(M1, fit1, "blob 1"), (M2, fit2, "blob 2")]:
            ii, jj = np.where(M)
            ax.scatter(jj+0.5, ii+0.5, s=6, c="dodgerblue", zorder=2)
            Xf, Yf = ellipse_points_image(*fitp, n=400)
            ax.plot(Xf, Yf, "r", lw=2, zorder=2)
        ax.set_aspect("equal", "box"); ax.axis("off")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"{img_path.stem}_panel{k+1}.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Ligne CSV
        px_to_mm = (Ak * args.mm_per_px) if args.mm_per_px else np.nan
        rows.append({
            "panel": k+1,
            "d1_px": d1, "d2_px": d2, "Ak_px": Ak,
            "Ak_mm": px_to_mm,
            "fit1_a": fit1[2], "fit1_b": fit1[3], "fit1_theta_deg": np.degrees(fit1[4]),
            "fit2_a": fit2[2], "fit2_b": fit2[3], "fit2_theta_deg": np.degrees(fit2[4]),
        })

        print(f"[ok] panel {k+1}: d1={d1:.2f}px d2={d2:.2f}px  Ak={Ak:.2f}px")

    # Moyenne finale
    mean_Ak = float(np.mean(Ak_list)) if Ak_list else float("nan")
    mean_mm = (mean_Ak * args.mm_per_px) if (args.mm_per_px and np.isfinite(mean_Ak)) else np.nan
    print(f"\n[final] mean(A_k) = {mean_Ak:.3f} px  ({mean_mm:.3f} mm)" if np.isfinite(mean_Ak) else "[final] no panels")

    # CSV
    header = ["panel","d1_px","d2_px","Ak_px","Ak_mm","fit1_a","fit1_b","fit1_theta_deg","fit2_a","fit2_b","fit2_theta_deg"]
    csv_path = out_dir / f"{img_path.stem}_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
        f.write(f"# mean_Ak_px,{mean_Ak}\n")
        if np.isfinite(mean_mm):
            f.write(f"# mean_Ak_mm,{mean_mm}\n")
    print(f"[ok] overlays + CSV -> {out_dir.resolve()}")

    return 0


# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
