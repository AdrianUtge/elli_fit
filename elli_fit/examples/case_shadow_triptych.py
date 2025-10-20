"""
===========================================================
Practical Case — Two-shadow diameter + ellipse fit (triptych)
===========================================================

But de ce script
-----------------
Traiter une image (grayscale) contenant N panneaux verticaux (par défaut 3),
et pour chaque panneau :
  1) détecter les deux ombres principales (méthode robuste par projection 1D),
  2) mesurer la séparation horizontale en pixels et convertir en millimètres
     (via --mm-per-px ou une calibration linéaire CSV sep_px,diam_mm),
  3) construire une ROI centrée entre les ombres,
  4) extraire le contour et ajuster une ellipse (RANSAC + LM optionnel) avec `elli_fit`,
  5) sauvegarder une image overlay par panneau + un CSV récapitulatif.

Points clés
-----------
- Détection des ombres par *projection horizontale* d'une bande verticale haute :
  on cherche les 2 pics les plus sombres (sur l'image normalisée & inversée).
- Fallback : si la projection échoue, sélection par composantes + scoring.
- Normalisation d’éclairage *flat-field* par panneau (NumPy-only) pour stabiliser le seuillage.
- Aucune modification de la librairie `elli_fit` n’est nécessaire.

Usage
-----
  python3 examples/case_shadow_triptych.py IMG.png \
      --mm-per-px 0.045 --refine --save-dir out_triptych_debug --debug-candidates

Options utiles
--------------
  --shadow-band 0.05,0.45     Bande verticale (fraction de hauteur) pour la projection
  --shadow-dark-q 0.80        Quantile de noirceur utilisé comme plancher (inversion)
  --min-sep-x 0.18            Séparation horizontale minimale (fraction de la largeur)
  --calib-csv calib.csv       Remplace --mm-per-px par une calibration linéaire

Auteur
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
    Lire PNG/JPG en niveaux de gris float32 dans [0,1] via matplotlib (sans dépendances).
    """
    arr = plt.imread(str(path))
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    arr = arr.astype(float)
    if arr.max() > 1.5:  # typiquement des uint8 -> [0..255]
        arr /= 255.0
    return arr


# --- Otsu threshold (sur [0,1]) ------------------------------------------

def otsu_thresh01(gray01: np.ndarray) -> float:
    g = np.clip(gray01, 0.0, 1.0)
    hist, _ = np.histogram(g.ravel(), bins=256, range=(0, 1))
    p = hist.astype(float) / max(1, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.linspace(0, 1, 256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return (k + 0.5) / 256.0


# --- Flat-field (normalisation d’éclairage locale) ------------------------

def _integral_image(A: np.ndarray) -> np.ndarray:
    return np.pad(A.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant")

def _box_filter_mean(A: np.ndarray, k: int) -> np.ndarray:
    H, W = A.shape
    II = _integral_image(A)
    r = int(max(1, k))
    y = np.arange(H)
    x = np.arange(W)
    Y0 = np.clip(y - r, 0, H - 1)
    Y1 = np.clip(y + r, 0, H - 1)
    X0 = np.clip(x - r, 0, W - 1)
    X1 = np.clip(x + r, 0, W - 1)
    Y0p, Y1p = Y0 + 1, Y1 + 1
    X0p, X1p = X0 + 1, X1 + 1
    Y0g, X0g = np.meshgrid(Y0p, X0p, indexing="ij")
    Y1g, X1g = np.meshgrid(Y1p, X1p, indexing="ij")
    area = (Y1g - Y0g + 1) * (X1g - X0g + 1)
    S = II[Y1g, X1g] - II[Y0g - 1, X1g] - II[Y1g, X0g - 1] + II[Y0g - 1, X0g - 1]
    return S / area

def normalize_flatfield(G: np.ndarray, win_frac: float = 0.12) -> np.ndarray:
    """
    G_norm = clip( (G / mean_local(G)), 0, 1 ) — fenêtre carrée ~ win_frac * min(H,W).
    """
    H, W = G.shape
    k = max(5, int(round(0.5 * win_frac * min(H, W))))
    base = _box_filter_mean(G, k)
    Gn = G / (base + 1e-6)
    Gn -= Gn.min()
    m = Gn.max()
    if m > 0:
        Gn /= m
    return np.clip(Gn, 0.0, 1.0)


# --- Connected components (4-connexe), features ---------------------------

def _connected_components_bool(B: np.ndarray):
    H, W = B.shape
    visited = np.zeros_like(B, dtype=bool)
    comps = []
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
                M = np.zeros_like(B, dtype=bool)
                for (ii, jj) in coords:
                    M[ii, jj] = True
                comps.append(M)
    return comps

def _component_features(Mc: np.ndarray, Ginv: np.ndarray):
    ii, jj = np.where(Mc)
    area = float(ii.size)
    if area == 0:
        return None
    x = jj.astype(float) + 0.5
    y = ii.astype(float) + 0.5
    cx, cy = float(x.mean()), float(y.mean())
    mean_dark = float(Ginv[ii, jj].mean())
    y0, y1 = int(ii.min()), int(ii.max())
    x0, x1 = int(jj.min()), int(jj.max())
    H, W = Mc.shape
    up = np.zeros_like(Mc); up[1:, :] = Mc[:-1, :]
    down = np.zeros_like(Mc); down[:-1, :] = Mc[1:, :]
    left = np.zeros_like(Mc); left[:, 1:] = Mc[:, :-1]
    right = np.zeros_like(Mc); right[:, :-1] = Mc[:, 1:]
    interior = up & down & left & right
    edge = Mc & (~interior)
    perim = float(edge.sum())
    compactness = (perim * perim) / (4.0 * np.pi * max(area, 1.0))
    return {
        "mask": Mc, "area": area, "mean_dark": mean_dark,
        "cx": cx, "cy": cy, "bbox": (x0, y0, x1, y1), "compactness": compactness
    }


# --- Détection par projection horizontale (robuste et simple) ------------

def _smooth_1d(v: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win) | 1)
    k = np.ones(win, dtype=float) / win
    return np.convolve(v, k, mode="same")

def find_two_shadows_by_projection(
    Gp_norm: np.ndarray,
    *,
    band=(0.05, 0.45),
    dark_q=0.80,
    min_sep_x_rel=0.18,
    smooth_frac=0.03,
    nms_frac=0.08
):
    """
    Détecte deux ombres via la projection horizontale (moyenne des intensités inversées)
    sur une bande verticale haute. Retourne (x1,y1,x2,y2, info).
    """
    H, W = Gp_norm.shape
    y0 = int(np.clip(band[0] * H, 0, H - 1))
    y1 = int(np.clip(band[1] * H, 1, H))
    if y1 <= y0 + 2:
        raise RuntimeError("projection band too thin")

    Ginv = 1.0 - Gp_norm
    strip = Ginv[y0:y1, :]
    prof = strip.mean(axis=0)
    floor = float(np.quantile(strip, dark_q))
    prof = np.where(prof >= floor, prof, 0.0)

    win = int(round(smooth_frac * W))
    prof_s = _smooth_1d(prof, win=max(3, win))
    nms = max(2, int(round(nms_frac * W)))

    idxs = []
    temp = prof_s.copy()
    for _ in range(2):
        j = int(np.argmax(temp))
        if temp[j] <= 0:
            break
        idxs.append(j)
        j0 = max(0, j - nms)
        j1 = min(W, j + nms + 1)
        temp[j0:j1] = 0.0

    if len(idxs) < 2:
        raise RuntimeError("projection: less than two peaks")

    idxs = sorted(idxs)
    x1, x2 = float(idxs[0]) + 0.5, float(idxs[1]) + 0.5
    if (x2 - x1) < (min_sep_x_rel * W):
        raise RuntimeError("projection: peaks too close in x")

    yc = 0.5 * (y0 + y1)
    return x1, yc, x2, yc, dict(band_px=(y0, y1), floor=floor, profile=prof_s)


# --- Sélection par composantes (fallback si projection échoue) ------------

def select_two_shadows(
    G: np.ndarray,
    B: np.ndarray,
    *,
    y_band=(0.05, 0.40),
    min_area_px=80,
    dark_quantile=0.80,
    min_sep_x_rel=0.18,
    k_candidates=12,
    w_area=1.0,
    w_dark=2.2,
    w_compact_penalty=0.8,
    w_pair_sep=3.0,
    w_pair_yalign=1.0,
    debug_ax=None
):
    H, W = G.shape
    Ginv = 1.0 - G
    dark_floor = float(np.quantile(Ginv, dark_quantile))
    comps = _connected_components_bool(B > 0)
    feats = []
    y0f, y1f = y_band
    y0, y1 = y0f * H, y1f * H

    for Mc in comps:
        f = _component_features(Mc, Ginv)
        if f is None:
            continue
        if f["area"] < min_area_px:
            continue
        if not (y0 <= f["cy"] <= y1):
            continue
        if f["mean_dark"] < dark_floor:
            continue
        feats.append(f)

    if len(feats) < 2:
        raise RuntimeError("Not enough candidates in band/contrast; relax constraints or band.")

    comp_scores = []
    for f in feats:
        s = w_area * np.sqrt(f["area"]) + w_dark * f["mean_dark"]
        s -= w_compact_penalty * max(0.0, 1.2 - min(f["compactness"], 1.2))
        comp_scores.append((s, f))
    comp_scores.sort(key=lambda t: t[0], reverse=True)
    cand = [f for s, f in comp_scores[:k_candidates]]

    best = None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            f1, f2 = cand[i], cand[j]
            sep_x_rel = abs(f2["cx"] - f1["cx"]) / max(1.0, W)
            if sep_x_rel < min_sep_x_rel:
                continue
            y_align = 1.0 - abs(f2["cy"] - f1["cy"]) / max(1.0, H)
            s_i = next(s for s, ff in comp_scores if ff is f1)
            s_j = next(s for s, ff in comp_scores if ff is f2)
            score_pair = 0.5 * (s_i + s_j) + w_pair_sep * sep_x_rel + w_pair_yalign * y_align
            if (best is None) or (score_pair > best[0]):
                best = (score_pair, f1, f2, sep_x_rel, y_align)

    if best is None:
        raise RuntimeError("No pair passes min horizontal separation; lower --min-sep-x.")

    score_pair, f1, f2, sep_x_rel, y_align = best

    if debug_ax is not None:
        for s, f in comp_scores[:k_candidates]:
            x0, y0b, x1, y1b = f["bbox"]
            debug_ax.add_patch(plt.Rectangle((x0, y0b), x1 - x0 + 1, y1b - y0b + 1,
                                             fill=False, ec="yellow", lw=1, ls="--"))
            debug_ax.text(f["cx"], f["cy"], f"{s:.2f}", color="yellow", ha="center", va="center", fontsize=8)
        debug_ax.axhline(y0, color="orange", ls=":", lw=1)
        debug_ax.axhline(y1, color="orange", ls=":", lw=1)

    return f1["mask"], f2["mask"], {
        "f1": f1, "f2": f2,
        "pair_score": float(score_pair),
        "sep_x_rel": float(sep_x_rel),
        "y_align": float(y_align),
        "dark_floor": float(dark_floor),
        "band_px": (float(y0), float(y1)),
    }

def auto_select_two_shadows(
    Gp, Bbin, *,
    prefer_y_band=(0.05, 0.40),
    dark_q_grid=(0.85, 0.80, 0.75, 0.70, 0.65),
    band_top_grid=(0.40, 0.50, 0.60),
    min_sep_grid=(0.22, 0.18, 0.15),
    min_area_px=90,
    debug_ax=None
):
    # strict d'abord
    try:
        b1, b2, info = select_two_shadows(
            G=Gp, B=Bbin, y_band=prefer_y_band, min_area_px=min_area_px,
            dark_quantile=dark_q_grid[0], min_sep_x_rel=min_sep_grid[0],
            k_candidates=12, w_area=1.0, w_dark=2.2, w_compact_penalty=0.8,
            w_pair_sep=3.0, w_pair_yalign=1.0, debug_ax=debug_ax,
        )
        params = dict(y_band=prefer_y_band, dark_q=dark_q_grid[0], min_sep=min_sep_grid[0])
        return b1, b2, info, params
    except Exception:
        pass

    # relaxation
    best = None
    for q in dark_q_grid:
        for top in band_top_grid:
            y_band = (prefer_y_band[0], top)
            for sep in min_sep_grid:
                try:
                    b1, b2, info = select_two_shadows(
                        G=Gp, B=Bbin, y_band=y_band, min_area_px=min_area_px,
                        dark_quantile=q, min_sep_x_rel=sep,
                        k_candidates=12, w_area=1.0, w_dark=2.2, w_compact_penalty=0.8,
                        w_pair_sep=3.0, w_pair_yalign=1.0, debug_ax=debug_ax,
                    )
                    score = info.get("pair_score", 0.0)
                    if (best is None) or (score > best[0]):
                        best = (score, (b1, b2, info, dict(y_band=y_band, dark_q=q, min_sep=sep)))
                except Exception:
                    continue
    if best is None:
        raise RuntimeError("Auto selector: no valid shadow pair after relaxation.")
    return best[1]


# --- Calibration utils ----------------------------------------------------

def load_linear_calibration(csv_path: Path):
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
    if calib_ab is not None:
        a, b = calib_ab
        return a * sep_px + b
    if mm_per_px is not None:
        return sep_px * mm_per_px
    return float("nan")


# --- Découpage en panneaux -----------------------------------------------

def split_equal_panels(G: np.ndarray, n: int):
    H, W = G.shape
    w = W // n
    panels = []
    for k in range(n):
        x0 = k * w
        x1 = (k + 1) * w if k < n - 1 else W
        panels.append(G[:, x0:x1])
    return panels


# --- Pipeline par panneau -------------------------------------------------

def process_panel(Gp: np.ndarray, args, panel_idx: int):
    """
    Détection des ombres (projection → fallback), mesure sép. + ellipse fit.
    Retourne (row_dict, pack_plot).
    """
    # Normalisation par panneau
    Gp = Gp.astype(float)
    if Gp.max() > 0:
        Gp /= Gp.max()
    Gp_norm = normalize_flatfield(Gp, win_frac=0.12)
    Ginv = 1.0 - Gp_norm

    # Binaire pour la suite (edges/ROI)
    thr = args.threshold if args.threshold is not None else otsu_thresh01(Ginv)
    B0 = (Ginv >= thr).astype(float)
    B = preprocess_mask(B0, threshold=0.5, clear_border=True, min_neighbors=1)

    # 1) Détection préférée : projection 1D (très robuste à ton setup)
    detected_via = "projection"
    try:
        y0f, y1f = (0.05, 0.45)
        if args.shadow_band:
            y0f, y1f = map(float, args.shadow_band.split(","))
        x1, y1, x2, y2, pj = find_two_shadows_by_projection(
            Gp_norm,
            band=(y0f, y1f),
            dark_q=args.shadow_dark_q,
            min_sep_x_rel=args.min_sep_x,
            smooth_frac=0.03,
            nms_frac=0.08,
        )
    except Exception:
        # 2) Fallback : sélection par composantes (auto-relax)
        detected_via = "components"
        blob1, blob2, info, chosen = auto_select_two_shadows(
            Gp, (B > 0.5),
            prefer_y_band=(0.05, 0.40),
            dark_q_grid=(0.85, 0.80, 0.75, 0.70, 0.65),
            band_top_grid=(0.40, 0.50, 0.60),
            min_sep_grid=(0.22, 0.18, 0.15),
            min_area_px=90,
            debug_ax=(plt.gca() if args.debug_candidates else None),
        )
        x1, y1 = info["f1"]["cx"], info["f1"]["cy"]
        x2, y2 = info["f2"]["cx"], info["f2"]["cy"]

    if not (np.isfinite(x1) and np.isfinite(x2)):
        raise RuntimeError("could not find two shadows")

    # Mesure séparation + conversion
    sep_px = abs(x2 - x1)
    diam_mm = pixels_to_mm(sep_px, mm_per_px=args.mm_per_px, calib_ab=args._calib)

    # ROI centrée entre les ombres
    cxr, cyr = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
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
    Broi = preprocess_mask(Broi0, threshold=0.5, clear_border=True, min_neighbors=0)

    # Edges → panneau coords
    xr, yr = edge_points_ordered(Broi)
    if len(xr) < 20:
        xr, yr = edge_points(Broi, threshold=0.5)
    if len(xr) < 6:
        raise RuntimeError("not enough edge points in ROI for ellipse fit")
    x = xr + x0
    y = yr + y0

    # Fit ellipse
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
        "detector": detected_via,
        "sep_px": float(sep_px),
        "diam_mm": float(diam_mm),
        "cx": float(cx), "cy": float(cy),
        "a": float(a), "b": float(b), "theta_deg": float(np.degrees(th)),
        "roi": (int(x0), int(y0), int(x1b), int(y1b)),
        "centroids": (float(x1), float(y1), float(x2), float(y2)),
    }, (Gp, x, y, cx, cy, a, b, th, (x0, y0, x1b, y1b), (x1, y1, x2, y2))


# --- Overlay par panneau --------------------------------------------------

def save_panel_overlay(out_dir: Path, base: str, k: int, pack, args):
    Gp, x, y, cx, cy, a, b, th, roi, cents = pack
    (x0, y0, x1b, y1b) = roi
    (x1, y1, x2, y2) = cents
    Xf, Yf = ellipse_points_image(cx, cy, a, b, th, n=args.samples)

    fig, ax = plt.subplots(figsize=(4.8, 6.4), facecolor="white")
    ax.set_facecolor("white")
    ax.imshow(Gp, cmap="gray", origin="upper", vmin=0, vmax=1)

    ax.add_patch(plt.Rectangle((x0, y0), x1b - x0, y1b - y0,
                               fill=False, ec="gold", lw=1.2, ls="--"))
    ax.scatter([x1, x2], [y1, y2], s=40, c="cyan", label="shadows")
    ax.plot([x1, x2], [y1, y2], "c--", lw=1)
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
    p.add_argument("image", help="Input image (PNG/JPG) with vertical panels.")
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
    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, save overlays here and a summary CSV.")
    # Détection ombres (projection)
    p.add_argument("--shadow-band", type=str, default="0.05,0.45",
                   help="Vertical band (fractions of H) for shadow projection, e.g. '0.05,0.45'.")
    p.add_argument("--shadow-dark-q", type=float, default=0.80,
                   help="Quantile on inverted gray as a minimal darkness in the projection [0..1].")
    p.add_argument("--min-sep-x", type=float, default=0.18,
                   help="Minimal horizontal separation between shadows as fraction of panel width.")
    p.add_argument("--debug-candidates", action="store_true",
                   help="If fallback triggers, overlay component candidates and scores.")
    return p


# --- Main ----------------------------------------------------------------

def main(argv=None):
    args = _build_argparser().parse_args(argv or sys.argv[1:])
    img_path = Path(args.image)
    G = imread_gray(img_path)
    base = img_path.stem

    # Calibration
    if args.calib_csv:
        args._calib = load_linear_calibration(Path(args.calib_csv))
    else:
        args._calib = None
        if args.mm_per_px is None:
            print("[warn] no calibration provided; diam_mm will be NaN")

    # Sortie
    out_dir = None
    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Découpage en panneaux
    panels = split_equal_panels(G, args.panels)

    rows = []
    packs = []
    for k, Gp in enumerate(panels):
        try:
            row, pack = process_panel(Gp, args, k)
            rows.append(row)
            packs.append(pack)
            print(f"[ok] panel {k+1} ({row['detector']}): sep={row['sep_px']:.1f}px  "
                  f"diam≈{row['diam_mm']:.3f}mm  a={row['a']:.2f}, b={row['b']:.2f}, "
                  f"θ={row['theta_deg']:.1f}°")
        except Exception as e:
            print(f"[fail] panel {k+1}: {e}")

    # Overlays + CSV
    if out_dir:
        for k, pack in enumerate(packs):
            _ = save_panel_overlay(out_dir, base, k, pack, args)

        if rows:
            header = ["panel", "detector", "sep_px", "diam_mm", "cx", "cy", "a", "b", "theta_deg"]
            out_csv = out_dir / f"{base}_summary.csv"
            with out_csv.open("w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for r in rows:
                    f.write(",".join(str(r[h]) for h in header) + "\n")
            print(f"[ok] overlays + summary -> {out_dir.resolve()}")
        else:
            print(f"[warn] no successful panels; overlays may be empty; no CSV written.")

    return 0


# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
