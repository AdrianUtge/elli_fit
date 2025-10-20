"""
===========================================================
Practical Case — Two-shadow diameter + ellipse fit (triptych)
===========================================================

But de ce script
-----------------
Ce script prend une image niveau de gris contenant plusieurs panneaux
(par défaut 3 panneaux verticaux côte à côte), et pour chaque panneau :
    1) détecte les deux ombres les plus sombres (les deux plus grands blobs),
    2) mesure la séparation horizontale en pixels et la convertit en mm
         (soit avec un facteur mm/pixel, soit avec une calibration linéaire CSV),
    3) construit une ROI centrée entre les deux ombres,
    4) extrait les contours puis ajuste une ellipse (RANSAC puis optionnellement
         LM pour raffiner),
    5) sauvegarde une image superposée (overlay) et un CSV sommaire.

Notes sur l'intention pédagogique
---------------------------------
Le but de cette version annotée est d'expliquer ligne par ligne les
choix d'implémentation, les appels à la bibliothèque `elli_fit` et les
opérations numpy/matplotlib pour qu'une personne étrangère au projet
comprenne le flux de données et les parties critiques.

Author
------
Adrian Utge Le Gall, 2025
"""

# --- Imports --------------------------------------------------------------

import sys
import argparse
from pathlib import Path

# NumPy est la bibliothèque principale pour le calcul numérique en Python.
import numpy as np

# Matplotlib est utilisé ici uniquement pour la lecture d'image et
# la génération d'overlays (PNG). Nous utilisons pyplot (plt) comme API
# de haut niveau pour tracer.
import matplotlib.pyplot as plt

# Import des fonctions principales de la librairie `elli_fit`.
# On importe explicitement les fonctions utilisées ci-dessous pour
# que le lecteur voie la dépendance claire vers le coeur algorithmique.
from elli_fit.core import (
    preprocess_mask,
    edge_points,
    edge_points_ordered,
    fit_ellipse_ransac,
    refine_ellipse_lm,
    ellipse_points_image,
)


# --- Illumination normalization (flat-field) ------------------------------

def _gauss_kernel1d(sigma: float):
    """Small 1D gaussian kernel for separable blur (NumPy-only)."""
    sigma = max(0.3, float(sigma))
    r = int(max(3, round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k

def _blur_gaussian_separable(G: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via 1D np.convolve on rows then cols."""
    k = _gauss_kernel1d(sigma)
    # rows
    R = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, G)
    # cols
    C = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, R)
    return C

# --- Illumination normalization (local mean via integral image) ----------

def _integral_image(A: np.ndarray) -> np.ndarray:
    """Integral image with zero-padded top/left border."""
    # shape -> (H+1, W+1)
    return np.pad(A.cumsum(0).cumsum(1), ((1,0),(1,0)), mode="constant")

def _box_filter_mean(A: np.ndarray, k: int) -> np.ndarray:
    """
    Local mean with square window size (2k+1). Output same shape as A.
    NumPy-only using integral image, O(1) per pixel.
    """
    H, W = A.shape
    II = _integral_image(A)
    r = k
    # coord clippées
    y0 = np.clip(np.arange(H) - r, 0, H-1)
    y1 = np.clip(np.arange(H) + r, 0, H-1)
    x0 = np.clip(np.arange(W) - r, 0, W-1)
    x1 = np.clip(np.arange(W) + r, 0, W-1)

    # vecteurs -> grilles
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")

    # passer en +1 pour matcher intégrale paddée
    Y0p, X0p = Y0+1, X0+1
    Y1p, X1p = Y1+1, X1+1

    area = (Y1 - Y0 + 1) * (X1 - X0 + 1)
    # somme fenêtre via intégrale
    S = II[Y1p, X1p] - II[Y0p-1, X1p] - II[Y1p, X0p-1] + II[Y0p-1, X0p-1]
    return S / area

def normalize_flatfield(G: np.ndarray, win_frac: float = 0.12) -> np.ndarray:
    """
    Flat-field robuste par panneau : divide by local mean in a (2k+1)x(2k+1) box.
    - win_frac : taille de fenêtre relative à min(H,W), ex: 0.12 ~ 12%
    - sortie clampée [0,1], même shape que G.
    """
    H, W = G.shape
    k = max(5, int(round(0.5 * win_frac * min(H, W))))  # demi-fenêtre
    base = _box_filter_mean(G, k)
    eps = 1e-6
    Gn = G / (base + eps)
    # rescale -> [0,1]
    Gn -= Gn.min()
    m = Gn.max()
    if m > 0:
        Gn /= m
    return np.clip(Gn, 0.0, 1.0)



# --- I/O (image) ----------------------------------------------------------

def imread_gray(path: Path) -> np.ndarray:
    """
    Read PNG/JPG as float grayscale in [0, 1] using matplotlib (no extra deps).
    """
    # plt.imread lit l'image en tableau numpy. Selon le format, on peut
    # obtenir une image 2D (grayscale) ou 3D (RGB ou RGBA).
    arr = plt.imread(str(path))

    # Si l'image est RGB(A) (trois ou quatre canaux), on convertit en
    # luminance perçue en utilisant les coefficients standards
    # (Rec. 709 / sRGB approximés) : Y = 0.2126 R + 0.7152 G + 0.0722 B
    if arr.ndim == 3:
        # Certains PNG ont un canal alpha ; si présent on l'ignore.
        if arr.shape[2] == 4:  # drop alpha if present
            arr = arr[..., :3]
        # Calcul de la luminance perçue (pondération des canaux)
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    # On travaille en float pour les opérations ultérieures.
    arr = arr.astype(float)

    # matplotlib peut retourner des images dans [0,1] (float) ou [0,255]
    # (uint8 converti en float). On détecte l'échelle et on normalise.
    if arr.max() > 1.5:  # si max > 1.5, on suppose l'échelle [0..255]
        arr /= 255.0
    # Retourne un array 2D float avec valeurs dans [0,1]
    return arr


# --- Thresholding (Otsu fallback) -----------------------------------------

def otsu_thresh01(gray01: np.ndarray) -> float:
    """
    Otsu threshold on [0,1] grayscale → returns scalar threshold in [0,1].
    """
    # On commence par contraindre les valeurs sur [0,1] pour éviter
    # des comportements surprenants si l'image contient des valeurs
    # aberrantes.
    g = np.clip(gray01, 0.0, 1.0)

    # Histogramme pour Otsu : 256 bins sur l'intervalle [0,1]
    # np.histogram retourne (counts, bin_edges)
    hist, _ = np.histogram(g.ravel(), bins=256, range=(0, 1))

    # Probabilités normalisées des intensités (somme = 1)
    p = hist.astype(float) / max(1, hist.sum())

    # omega(k) = somme_{i<=k} p(i) -> probabilité cumulative
    omega = np.cumsum(p)

    # mu(k) = somme_{i<=k} p(i) * intensity(i) -> moment cumulatif
    mu = np.cumsum(p * np.linspace(0, 1, 256))
    mu_t = mu[-1]  # moment total

    # Variance inter-classes (Otsu) : on cherche k qui maximise sigma_b^2
    # ajout d'un petit terme numérique pour éviter division par zéro
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)

    # np.nanargmax renvoie l'indice du maximum en ignorant les NaN.
    # En pratique, si l'image est constante, sigma_b2 peut être NaN partout ;
    # ici on suppose que np.nanargmax renvoie quelque chose raisonnable,
    # mais un durcissement serait de détecter le cas constant et renvoyer
    # 0.5.
    k = int(np.nanargmax(sigma_b2))

    # On convertit l'indice de bin en seuil sur [0,1] en prenant le centre
    # du bin : (k + 0.5)/256
    return (k + 0.5) / 256.0


# --- Connected components (largest two blobs, 4-neighborhood) -------------

# --- Robust shadow selection ---------------------------------------------

def _connected_components_bool(B: np.ndarray):
    """
    Retourne une liste de masques booléens, un par composant connexe
    (voisinage 4-connexe). Cette implémentation utilise une BFS (pile)
    écrite en Python pur.

    Remarque sur les performances :
    - Cette fonction est simple et portable (pas de dépendance), mais
      peut être lente pour de grandes images. Si vous acceptez scipy,
      `scipy.ndimage.label` est beaucoup plus rapide.

    Entrée :
      B : tableau booléen 2D indiquant les pixels foreground.
    Sortie :
      liste de masques booléens de même taille que B.
    """
    H, W = B.shape
    # visited : masque des pixels déjà parcourus
    visited = np.zeros_like(B, dtype=bool)
    comps = []

    # On parcourt chaque pixel ; lorsqu'on trouve un pixel foreground non
    # visité, on lance une exploration en profondeur / largeur (ici on
    # utilise une pile q pour BFS/DFS) pour récupérer tout le composant.
    for i in range(H):
        for j in range(W):
            # Condition d'entrée : pixel foreground et non visité
            if B[i, j] and not visited[i, j]:
                q = [(i, j)]
                visited[i, j] = True
                coords = [(i, j)]  # liste des coordonnées (i,j) du composant

                # exploration du composant
                while q:
                    ci, cj = q.pop()
                    # on teste les 4 voisins (haut, bas, gauche, droite)
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = ci + di, cj + dj
                        # vérification des bornes et du fait que le voisin
                        # appartient au foreground et n'a pas encore été visité
                        if 0 <= ni < H and 0 <= nj < W and B[ni, nj] and not visited[ni, nj]:
                            visited[ni, nj] = True
                            q.append((ni, nj))
                            coords.append((ni, nj))

                # Construction du masque booléen du composant à partir des
                # coordonnées collectées. On pourrait optimiser en utilisant
                # des indexations avancées, mais la clarté prime ici.
                M = np.zeros_like(B, dtype=bool)
                for (ii, jj) in coords:
                    M[ii, jj] = True
                comps.append(M)

    return comps


def _component_features(Mc: np.ndarray, Ginv: np.ndarray):
    """
    Compute features for a single component mask:
      - area, mean_dark, centroid (x,y), bbox, compactness (optional)
    """
    # ii, jj sont les indices (ligne, colonne) des pixels du composant
    ii, jj = np.where(Mc)

    # Surface en pixels
    area = float(ii.size)
    if area == 0:
        return None

    # Les coordonnées image sont ici définies au centre du pixel :
    # x = colonne + 0.5, y = ligne + 0.5
    x = jj.astype(float) + 0.5
    y = ii.astype(float) + 0.5
    cx = float(x.mean())
    cy = float(y.mean())

    # mean_dark est calculé sur l'image inversée Ginv (plus grand -> plus sombre)
    mean_dark = float(Ginv[ii, jj].mean())  # plus c'est grand, plus c'est sombre

    # bbox : boîte englobante minimale (x0,y0)-(x1,y1) en coordonnées entières
    y0, y1 = int(ii.min()), int(ii.max())
    x0, x1 = int(jj.min()), int(jj.max())
    bbox = (x0, y0, x1, y1)

    # Estimation approximative du périmètre via le voisinage 4-connexe :
    # on considère un pixel intérieur s'il a ses 4 voisins également à 1.
    H, W = Mc.shape
    up = np.zeros_like(Mc); up[1:,  :]  = Mc[:-1, :]
    down = np.zeros_like(Mc); down[:-1, :] = Mc[1:, :]
    left = np.zeros_like(Mc); left[:, 1:] = Mc[:, :-1]
    right = np.zeros_like(Mc); right[:, :-1] = Mc[:, 1:]
    interior = up & down & left & right
    edge = Mc & (~interior)
    perim = float(edge.sum())

    # Compactness : (P^2) / (4*pi*A) égale 1 pour un disque parfait,
    # plus grande pour des formes moins compactes.
    compactness = (perim * perim) / (4.0 * np.pi * max(area, 1.0))

    return {
        "mask": Mc,
        "area": area,
        "mean_dark": mean_dark,
        "cx": cx,
        "cy": cy,
        "bbox": bbox,
        "compactness": compactness,
    }

# --- Robust shadow selection (band + contrast + min separation) ----------
# --- Auto-relaxing selector wrapper --------------------------------------

def auto_select_two_shadows(Gp, Bbin, *, prefer_y_band=(0.05, 0.40),
                            dark_q_grid=(0.85, 0.80, 0.75, 0.70, 0.65),
                            band_top_grid=(0.40, 0.50, 0.60),
                            min_sep_grid=(0.22, 0.18, 0.15),
                            min_area_px=90,
                            debug_ax=None):
    """
    Try strict params first; if no pair found, relax darkness/band/separation
    over a small grid and return the first successful pair with the highest
    pair score. Returns (blob1, blob2, info) and chosen 'params' dict.
    """
    # 1) try preferred strict setting
    try:
        b1, b2, info = select_two_shadows(
            G=Gp, B=Bbin,
            y_band=prefer_y_band,
            min_area_px=min_area_px,
            dark_quantile=dark_q_grid[0],
            min_sep_x_rel=min_sep_grid[0],
            k_candidates=12,
            w_area=1.0, w_dark=2.2, w_compact_penalty=0.8,
            w_pair_sep=3.0, w_pair_yalign=1.0,
            debug_ax=debug_ax,
        )
        params = dict(y_band=prefer_y_band, dark_q=dark_q_grid[0], min_sep=min_sep_grid[0])
        return b1, b2, info, params
    except Exception:
        pass

    # 2) grid search (small, fast)
    best = None
    H, W = Gp.shape
    for q in dark_q_grid:
        for top in band_top_grid:
            y_band = (prefer_y_band[0], top)
            for sep in min_sep_grid:
                try:
                    b1, b2, info = select_two_shadows(
                        G=Gp, B=Bbin,
                        y_band=y_band,
                        min_area_px=min_area_px,
                        dark_quantile=q,
                        min_sep_x_rel=sep,
                        k_candidates=12,
                        w_area=1.0, w_dark=2.2, w_compact_penalty=0.8,
                        w_pair_sep=3.0, w_pair_yalign=1.0,
                        debug_ax=debug_ax,
                    )
                    score = info.get("pair_score", 0.0)
                    if (best is None) or (score > best[0]):
                        best = (score, (b1, b2, info, dict(y_band=y_band, dark_q=q, min_sep=sep)))
                except Exception:
                    continue

    if best is None:
        raise RuntimeError("Auto selector: no valid shadow pair after relaxation.")
    return best[1]

def select_two_shadows(
    G: np.ndarray,
    B: np.ndarray,
    *,
    y_band=(0.05, 0.40),        # strong: top 5% to 40% of height
    min_area_px=80,
    dark_quantile=0.80,         # adaptive contrast floor on Ginv
    min_sep_x_rel=0.18,         # min horizontal separation as fraction of width
    k_candidates=12,
    w_area=1.0,
    w_dark=2.2,
    w_compact_penalty=0.8,
    w_pair_sep=3.0,
    w_pair_yalign=1.0,
    debug_ax=None
):
    """
    Select the two shadow components using hard pre-filters + pair scoring.

    Hard pre-filters (reject-early):
      - y inside [y0,y1] band (relative to H)
      - area >= min_area_px
      - mean_dark >= quantile(dark_quantile) of Ginv
    Pair constraint:
      - |Δx|/W >= min_sep_x_rel

    Then score remaining components and pick the best pair.
    """
    H, W = G.shape

    # On travaille sur l'image inversée Ginv où les ombres (sombres)
    # deviennent des régions de grande valeur : cela simplifie la
    # mesure de "mean_dark" (moyenne sur Ginv, plus grand -> plus sombre)
    Ginv = 1.0 - G

    # Seuil adaptatif global (quantile) pour rejeter les régions trop
    # claires ; utile en présence de vignettage.
    dark_floor = float(np.quantile(Ginv, dark_quantile))

    # Extraction des composants connexes (masques booléens) 4-connexes.
    # On passe B > 0 pour convertir en bool si B est float (0/1).
    comps = _connected_components_bool(B > 0)
    feats = []

    # Convertit la bande verticale relative en pixels [y0,y1]
    y0f, y1f = y_band
    y0, y1 = y0f * H, y1f * H

    # Boucle sur chaque composant pour calculer des caractéristiques
    # et appliquer des filtres d'exclusion (area, position verticale, contraste)
    for Mc in comps:
        f = _component_features(Mc, Ginv)
        if f is None:
            continue
        # filtre par surface minimale
        if f["area"] < min_area_px:
            continue
        # filtre par position verticale (on s'attend aux ombres dans
        # une bande supérieure du panneau)
        if not (y0 <= f["cy"] <= y1):
            continue
        # filtre par niveau sombre minimal (robuste au fond non uniforme)
        if f["mean_dark"] < dark_floor:
            continue
        feats.append(f)

    if len(feats) < 2:
        # Pas assez de candidats ; l'appelant peut choisir de détendre
        # les contraintes ou d'augmenter la bande.
        raise RuntimeError("Not enough candidates in band/contrast; relax constraints or band.")

    # Calcul d'un score individuel pour trier/pondérer les composants.
    # La formule combine la racine de la surface (stabilise l'échelle),
    # la profondeur moyenne (mean_dark) et pénalise les formes peu
    # compactes.
    comp_scores = []
    for f in feats:
        s = w_area * np.sqrt(f["area"]) + w_dark * f["mean_dark"]
        # penalité : si la compactness est < 1.2 on applique une pénalité
        s -= w_compact_penalty * max(0.0, 1.2 - min(f["compactness"], 1.2))
        comp_scores.append((s, f))

    # Tri décroissant par score
    comp_scores.sort(key=lambda t: t[0], reverse=True)

    # Sélection des meilleurs candidats (k_candidates) : on prend la
    # composante (f) issue du tuple (score, f) récupérer par la
    # compréhension de liste suivante :
    #   cand = [f for s, f in comp_scores[:k_candidates]]
    # Explication de la compréhension : on parcourt chaque tuple (s,f)
    # dans la liste des k premiers éléments et on prend seulement f.
    cand = [f for s, f in comp_scores[:k_candidates]]

    # Recherche de la meilleure paire parmi les candidats selon une
    # métrique qui combine la séparation horizontale normalisée et
    # l'alignement vertical.
    best = None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            f1, f2 = cand[i], cand[j]
            # séparation relative en x (normalisée par la largeur W)
            sep_x_rel = abs(f2["cx"] - f1["cx"]) / max(1.0, W)
            if sep_x_rel < min_sep_x_rel:
                # paire trop rapprochée horizontalement, on la rejette
                continue
            # alignement vertical : 1.0 si mêmes y, décroit quand
            # l'éloignement vertical augmente
            y_align = 1.0 - abs(f2["cy"] - f1["cy"]) / max(1.0, H)

            # récupération des scores individuels (on utilise next() pour
            # retrouver le score associé à un composant f donnée la
            # structure comp_scores)
            s_i = next(s for s, ff in comp_scores if ff is f1)
            s_j = next(s for s, ff in comp_scores if ff is f2)

            # score global de la paire : moyenne des scores individuels
            # + terme favorisant la séparation + terme favorisant
            # l'alignement vertical
            score_pair = 0.5 * (s_i + s_j) + w_pair_sep * sep_x_rel + w_pair_yalign * y_align
            if (best is None) or (score_pair > best[0]):
                best = (score_pair, f1, f2, sep_x_rel, y_align)

    if best is None:
        raise RuntimeError("No pair passes min horizontal separation; lower --min-sep-x.")

    score_pair, f1, f2, sep_x_rel, y_align = best

    # Si l'utilisateur demande un axe de debug, on dessine les
    # boîtes englobantes et les scores pour aider au diagnostic.
    if debug_ax is not None:
        # draw all candidates with scores
        for s, f in comp_scores[:k_candidates]:
            x0, y0b, x1, y1b = f["bbox"]
            # Rectangle : (x0,y0) position, largeur = x1-x0+1, hauteur = y1-y0+1
            debug_ax.add_patch(plt.Rectangle((x0, y0b), x1 - x0 + 1, y1b - y0b + 1,
                                             fill=False, ec="yellow", lw=1, ls="--"))
            # annotation du score au centre du composant
            debug_ax.text(f["cx"], f["cy"], f"{s:.2f}", color="yellow",
                          ha="center", va="center", fontsize=8)
        # draw band
        debug_ax.axhline(y0, color="orange", ls=":", lw=1)
        debug_ax.axhline(y1, color="orange", ls=":", lw=1)

    # On retourne les masques des deux meilleurs composants et des
    # informations complémentaires sur le choix.
    return f1["mask"], f2["mask"], {
        "f1": f1,
        "f2": f2,
        "pair_score": float(score_pair),
        "sep_x_rel": float(sep_x_rel),
        "y_align": float(y_align),
        "dark_floor": float(dark_floor),
        "band_px": (float(y0), float(y1)),
    }
    
    
    
    
# --- Auto-relaxing selector wrapper --------------------------------------

def auto_select_two_shadows(Gp, Bbin, *, prefer_y_band=(0.05, 0.40),
                            dark_q_grid=(0.85, 0.80, 0.75, 0.70, 0.65),
                            band_top_grid=(0.40, 0.50, 0.60),
                            min_sep_grid=(0.22, 0.18, 0.15),
                            min_area_px=90,
                            debug_ax=None):
    """
    Try strict params first; if no pair found, relax darkness/band/separation
    over a small grid and return the first successful pair with the highest
    pair score. Returns (blob1, blob2, info) and chosen 'params' dict.
    """
    # 1) try preferred strict setting
    try:
        b1, b2, info = select_two_shadows(
            G=Gp, B=Bbin,
            y_band=prefer_y_band,
            min_area_px=min_area_px,
            dark_quantile=dark_q_grid[0],
            min_sep_x_rel=min_sep_grid[0],
            k_candidates=12,
            w_area=1.0, w_dark=2.2, w_compact_penalty=0.8,
            w_pair_sep=3.0, w_pair_yalign=1.0,
            debug_ax=debug_ax,
        )
        params = dict(y_band=prefer_y_band, dark_q=dark_q_grid[0], min_sep=min_sep_grid[0])
        return b1, b2, info, params
    except Exception:
        pass

    # 2) grid search (small, fast)
    best = None
    H, W = Gp.shape
    for q in dark_q_grid:
        for top in band_top_grid:
            y_band = (prefer_y_band[0], top)
            for sep in min_sep_grid:
                try:
                    b1, b2, info = select_two_shadows(
                        G=Gp, B=Bbin,
                        y_band=y_band,
                        min_area_px=min_area_px,
                        dark_quantile=q,
                        min_sep_x_rel=sep,
                        k_candidates=12,
                        w_area=1.0, w_dark=2.2, w_compact_penalty=0.8,
                        w_pair_sep=3.0, w_pair_yalign=1.0,
                        debug_ax=debug_ax,
                    )
                    score = info.get("pair_score", 0.0)
                    if (best is None) or (score > best[0]):
                        best = (score, (b1, b2, info, dict(y_band=y_band, dark_q=q, min_sep=sep)))
                except Exception:
                    continue

    if best is None:
        raise RuntimeError("Auto selector: no valid shadow pair after relaxation.")
    return best[1]


# --- Calibration ----------------------------------------------------------

def load_linear_calibration(csv_path: Path):
    """
    CSV with two columns: sep_px, diam_mm. Fit diam = alpha*sep + beta.
    Returns (alpha, beta).
    """
    # np.genfromtxt lit le CSV et retourne un tableau numpy. On autorise
    # des commentaires commençant par '#'.
    data = np.genfromtxt(str(csv_path), delimiter=",", dtype=float, comments="#")

    # Cas particulier : si le CSV ne contient qu'une seule ligne,
    # genfromtxt retourne un vecteur 1D. On le remet en matrice n×2.
    if data.ndim == 1:
        data = data.reshape(-1, 2)

    if data.shape[1] != 2:
        raise ValueError("Calibration CSV must have 2 columns: sep_px,diam_mm")

    sep = data[:, 0]
    dia = data[:, 1]

    # Résolution du problème linéaire en moindres carrés : dia ≈ alpha*sep + beta
    # on construit la matrice A = [sep, 1] et on résout pour [alpha, beta]
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
    # largeur en pixels d'un panneau (division entière)
    w = W // n
    panels = []
    for k in range(n):
        # calcul des bornes inclusives/exclusives en x pour chaque panneau
        x0 = k * w
        # pour le dernier panneau on prend jusqu'à la largeur totale W
        x1 = (k + 1) * w if k < n - 1 else W
        # on découpe la matrice G en colonnes [x0:x1]
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

    # Invert and per-panel illumination normalization
    Gp = Gp.astype(float)
    Gp /= (Gp.max() + 1e-12)  # sécurité
    Gp_norm = normalize_flatfield(Gp, win_frac=0.12)  # même shape (H,W) pour CE panneau
    Ginv = 1.0 - Gp_norm

    # Threshold (Otsu si non fourni) sur l'image normalisée
    thr = args.threshold if args.threshold is not None else otsu_thresh01(Ginv)
    B0 = (Ginv >= thr).astype(float)

    # Nettoyage binaire
    B = preprocess_mask(B0, threshold=0.5, clear_border=True, min_neighbors=1)


    # Clean binary (keep thin strokes; clear border)
    B = preprocess_mask(B0, threshold=0.5, clear_border=True, min_neighbors=1)

    # Robust auto-selection of the two shadows (relax if needed)
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


    if not np.isfinite(x1) or not np.isfinite(x2):
        raise RuntimeError("could not find two shadows; adjust threshold")

    # Distance horizontale en pixels entre les deux centres (séparation)
    sep_px = abs(x2 - x1)
    # Conversion en mm via la calibration (fonction utilitaire)
    diam_mm = pixels_to_mm(sep_px, mm_per_px=args.mm_per_px, calib_ab=args._calib)

    # Construction d'une ROI carrée centrée entre les deux ombres.
    cxr = 0.5 * (x1 + x2)
    cyr = 0.5 * (y1 + y2)
    # largeur approximée de la ROI = 0.9 * séparation, mais au moins 20 px
    w_roi = max(20.0, 0.9 * sep_px)
    h_roi = w_roi
    H, W = Gp.shape
    x0 = max(0, int(cxr - w_roi / 2))
    x1b = min(W, int(cxr + w_roi / 2))
    y0 = max(0, int(cyr - h_roi / 2))
    y1b = min(H, int(cyr + h_roi / 2))

    # Extraction de la ROI et seuillage local
    ROI = Gp[y0:y1b, x0:x1b]
    ROIinv = 1.0 - ROI
    thr_roi = args.threshold if args.threshold is not None else otsu_thresh01(ROIinv)
    Broi0 = (ROIinv >= thr_roi).astype(float)
    # Préserver des arêtes fines : min_neighbors=0 passe les pixels isolés
    Broi = preprocess_mask(Broi0, threshold=0.5, clear_border=True, min_neighbors=0)

    # Extraction des points de bord ordonnés (edge_points_ordered) :
    # retourne deux vecteurs xr, yr (coordonnées locales à la ROI)
    xr, yr = edge_points_ordered(Broi)
    # si peu de points ordonnés, on bascule sur la version non ordonnée
    if len(xr) < 20:
        # edge_points peut accepter un paramètre threshold selon l'API
        xr, yr = edge_points(Broi, threshold=0.5)
    if len(xr) < 6:
        raise RuntimeError("not enough edge points in ROI for ellipse fit")

    # On remet les coordonnées dans le repère du panneau en ajoutant
    # l'origine x0,y0 de la ROI.
    x = xr + x0
    y = yr + y0

    # Ajustement robuste par RANSAC : renvoie (cx,cy,a,b,theta)
    cx, cy, a, b, th = fit_ellipse_ransac(
        x, y,
        iters=args.ransac_iters,
        sample_size=args.ransac_sample,
        inlier_thresh=args.ransac_thresh,
        min_inliers=max(20, len(x) // 3),
    )

    # Optionnel : raffinement non-linéaire Levenberg-Marquardt
    if args.refine:
        cx, cy, a, b, th = refine_ellipse_lm(x, y, cx, cy, a, b, th, iters=25, lam=1e-2)

    # On retourne un dictionnaire récapitulatif et un "pack" utile pour
    # tracer l'overlay (image + points + ellipse)
    return {
        "panel": panel_idx,
        "sep_px": float(sep_px),
        "diam_mm": float(diam_mm),
        "cx": float(cx),
        "cy": float(cy),
        "a": float(a),
        "b": float(b),
        "theta_deg": float(np.degrees(th)),
        "roi": (int(x0), int(y0), int(x1b), int(y1b)),
        "centroids": (float(x1), float(y1), float(x2), float(y2)),
    }, (Gp, x, y, cx, cy, a, b, th, (x0, y0, x1b, y1b), (x1, y1, x2, y2))


# --- Overlay per panel ----------------------------------------------------

def save_panel_overlay(out_dir: Path, base: str, k: int, pack, args):
    """
    Save a PNG overlay for a processed panel.
    """
    # Dépaquetage des informations utiles pour le tracé
    Gp, x, y, cx, cy, a, b, th, roi, cents = pack
    (x0, y0, x1b, y1b) = roi
    (x1, y1, x2, y2) = cents

    # ellipse_points_image (de la librairie) génère deux vecteurs Xf,Yf
    # correspondant aux coordonnées de points échantillonnés sur
    # l'ellipse paramétrée par (cx,cy,a,b,th). n=args.samples contrôle
    # la densité de l'échantillonnage.
    Xf, Yf = ellipse_points_image(cx, cy, a, b, th, n=args.samples)

    # Création d'une figure matplotlib ; figsize choisi pour un rendu
    # lisible sur l'overlay.
    fig, ax = plt.subplots(figsize=(4.6, 6.2), facecolor="white")
    ax.set_facecolor("white")

    # Affiche l'image en niveaux de gris ; origin='upper' aligne la
    # convention image (ligne 0 en haut) avec matplotlib.
    ax.imshow(Gp, cmap="gray", origin="upper", vmin=0, vmax=1)

    # Rectangle de la ROI : on utilise plt.Rectangle pour tracer la
    # boîte englobante (non remplie).
    ax.add_patch(plt.Rectangle((x0, y0), x1b - x0, y1b - y0,
                               fill=False, ec="gold", lw=1.2, ls="--"))

    # Centroides des ombres et segment de liaison
    ax.scatter([x1, x2], [y1, y2], s=36, c="cyan", label="shadows")
    ax.plot([x1, x2], [y1, y2], "c--", lw=1)

    # Points d'arête et ellipse ajustée
    ax.scatter(x, y, s=8, c="dodgerblue", label=f"edge points (n={len(x)})")
    ax.plot(Xf, Yf, "r", lw=2, label=f"fit θ={np.degrees(th):.1f}°")

    # Mise en forme finale : aspect égal pour éviter la distortion
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    # Création du dossier de sortie si nécessaire
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base}_panel{k+1}.png"
    # Sauvegarde PNG : bbox_inches='tight' enlève les marges inutiles
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
    # Par défaut None est plus explicite que '', et permet de tester
    # facilement si l'utilisateur a demandé une sauvegarde.
    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, save overlays here and a summary CSV.")
    p.add_argument("--shadow-band", type=str, default="0.05,0.40",
               help="Preferred vertical band (fractions of H) for shadows, e.g. '0.05,0.40'.")
    p.add_argument("--shadow-dark-q", type=float, default=0.80,
               help="Quantile on inverted gray used as minimal darkness for candidates [0..1].")
    p.add_argument("--min-sep-x", type=float, default=0.18,
               help="Minimal horizontal separation between shadows as fraction of panel width.")
    p.add_argument("--debug-candidates", action="store_true",
               help="Overlay candidate components and their scores for diagnostics.")

    return p


# --- Main ----------------------------------------------------------------

def main(argv=None):
    args = _build_argparser().parse_args(argv or sys.argv[1:])
    img_path = Path(args.image)
    G = imread_gray(img_path)
    base = img_path.stem

    # Parsing des paramètres utilisateur : calibration
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    else:
        args._calib = None
        if args.mm_per_px is None:
            print("[warn] no calibration provided; diam_mm will be NaN")

    # Panels (equal width split)
    panels = split_equal_panels(G, args.panels)

    rows = []
    packs = []
    # création du répertoire de sortie si demandé
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
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

    # Sauvegarde des overlays et du CSV résumant les mesures
    if args.save_dir:
        out_dir = Path(args.save_dir)
        for k, pack in enumerate(packs):
            _ = save_panel_overlay(out_dir, base, k, pack, args)

        header = ["panel", "sep_px", "diam_mm", "cx", "cy", "a", "b", "theta_deg"]
        out_csv = out_dir / f"{base}_summary.csv"
        # écriture simple du CSV en format texte
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                # pour chaque enregistrement, on écrit les colonnes dans l'ordre
                f.write(",".join(str(r[h]) for h in header) + "\n")
        print(f"[ok] overlays + summary -> {out_dir.resolve()}")

    return 0


# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
