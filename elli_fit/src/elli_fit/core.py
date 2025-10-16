"""
===========================================================
elli_fit.core — compact ellipse fitting core (NumPy-only)
===========================================================

Implements the essential computational parts of the library:
  - preprocess_mask()      : binarize & clean simple artifacts
  - edge_points()          : extract edge pixel centers (unordered)
  - edge_points_ordered()  : ordered contour tracing (8-neighbors)
  - fit_ellipse_ls()       : direct least-squares ellipse fit (Fitzgibbon)
  - fit_ellipse_ransac()   : robust fit via RANSAC over fit_ellipse_ls
  - ellipse_points()       : sample points on a fitted ellipse

Design goals
------------
- Minimal, clear, and robust (NumPy only)
- Hartley normalization + tiny ridge + pinv fallback
- Orientation normalized (a ≥ b, θ ∈ [-π/2, π/2))
- Automatic disambiguation between θ and θ+π/2 (pick best)

Author
------
Adrian Utge Le Gall, 2025
"""

# --- Imports --------------------------------------------------------------

import numpy as np


# --- Preprocess / Cleaning -----------------------------------------------

def preprocess_mask(M: np.ndarray, threshold: float = 0.5,
                    clear_border: bool = True,
                    min_neighbors: int = 2) -> np.ndarray:
    """
    Binarize then clean simple artifacts:
      - optional: clear borders (first/last row & col)
      - remove isolated pixels / tiny spurs by keeping pixels that have
        >= min_neighbors in their 3x3 neighborhood (excluding self)

    Parameters
    ----------
    M : np.ndarray
        2D numeric array.
    threshold : float
        Values > threshold are foreground.
    clear_border : bool
        If True, zero out first/last row & column (useful for CSVs with ramps).
    min_neighbors : int
        Minimum number of 3x3 neighbors (0..8) required to keep a pixel.

    Returns
    -------
    B : np.ndarray
        Cleaned binary mask in {0.0, 1.0}.
    """
    B = (M > threshold).astype(np.uint8)
    if B.ndim != 2:
        raise ValueError("preprocess_mask expects a 2D array")

    if clear_border:
        B[0, :] = B[-1, :] = 0
        B[:, 0] = B[:, -1] = 0

    if min_neighbors > 0:
        # Count neighbors using padding and shifts (NumPy only).
        H, W = B.shape
        P = np.pad(B, 1, mode="constant", constant_values=0)
        cnt = np.zeros_like(B, dtype=np.uint8)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                cnt += P[1+di:1+di+H, 1+dj:1+dj+W]
        # Drop pixels with too few neighbors
        B[(B == 1) & (cnt < min_neighbors)] = 0

    return B.astype(float)


# --- Edge Extraction (4-neighbors, unordered) -----------------------------

def edge_points(M: np.ndarray, threshold: float = 0.5):
    """
    Extract edge pixel centers (x, y) from a binary grid (unordered list).

    Parameters
    ----------
    M : np.ndarray
        2D numeric array (0–1 image or thresholdable matrix).
    threshold : float
        Values > threshold are considered foreground.

    Returns
    -------
    x, y : np.ndarray
        Coordinates of pixel centers marking the boundary.
    """
    B = (M > threshold).astype(np.uint8)
    up    = np.zeros_like(B); up[1:,  :]  = B[:-1, :]
    down  = np.zeros_like(B); down[:-1,:] = B[1:,  :]
    left  = np.zeros_like(B); left[:, 1:] = B[:, :-1]
    right = np.zeros_like(B); right[:, :-1]= B[:, 1:]
    interior = up * down * left * right
    edge = (B == 1) & (interior == 0)
    i, j = np.where(edge)
    return j.astype(float) + 0.5, i.astype(float) + 0.5


# --- Ordered contour (Moore-Neighbor tracing, 8-neighbors) ---------------

def edge_points_ordered(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Trace a single contour and return ordered (x,y) pixel centers.
    If multiple contours exist, it picks the first it finds.

    Parameters
    ----------
    M : np.ndarray
        Binary mask (values in {0,1} or booleans).

    Returns
    -------
    x, y : np.ndarray
        Ordered boundary coordinates (may be empty if no contour).
    """
    B = (M > 0).astype(np.uint8)
    if B.ndim != 2:
        raise ValueError("edge_points_ordered expects a 2D array")
    H, W = B.shape

    # Find a starting edge pixel (4-neighbor definition of edge)
    start = None
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if B[i, j] == 1:
                if (B[i-1, j] * B[i+1, j] * B[i, j-1] * B[i, j+1]) == 0:
                    start = (i, j)
                    break
        if start:
            break
    if start is None:
        return np.array([]), np.array([])

    # Moore-Neighbor tracing (8-neighborhood, clockwise)
    di = [-1, -1,  0,  1,  1,  1,  0, -1]
    dj = [ 0,  1,  1,  1,  0, -1, -1, -1]
    i, j = start
    prev_dir = 6  # start "coming from" west (arbitrary)
    contour_i, contour_j = [], []

    for _ in range(20_000):  # hard safety cap
        contour_i.append(i)
        contour_j.append(j)
        # start search from prev_dir-2 (clockwise)
        k0 = (prev_dir + 6) % 8
        found = False
        for k in range(8):
            kk = (k0 + k) % 8
            ni, nj = i + di[kk], j + dj[kk]
            if 0 <= ni < H and 0 <= nj < W and B[ni, nj] == 1:
                i, j = ni, nj
                prev_dir = kk
                found = True
                break
        if not found or (i, j) == start:
            break

    x = np.asarray(contour_j, float) + 0.5
    y = np.asarray(contour_i, float) + 0.5
    return x, y


# --- Utilities ------------------------------------------------------------

def _normalize_axes_angle(a: float, b: float, theta: float):
    """
    Ensure a ≥ b and wrap θ into [-π/2, π/2).
    """
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0
    theta = (theta + np.pi / 2.0) % np.pi - np.pi / 2.0
    return a, b, theta


def _best_orientation(x, y, cx, cy, a, b, theta):
    """
    Compare residuals for (a,b,θ) vs (b,a,θ+π/2) and return the best.
    This resolves the 90° ambiguity frequently seen with ellipse fits.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)

    # Candidate 1: (a, b, theta)
    c, s = np.cos(theta), np.sin(theta)
    xr =  c*(x - cx) + s*(y - cy)
    yr = -s*(x - cx) + c*(y - cy)
    r1 = np.mean(((xr / a)**2 + (yr / b)**2 - 1.0)**2)

    # Candidate 2: (b, a, theta + π/2)
    theta2 = theta + np.pi / 2.0
    c2, s2 = np.cos(theta2), np.sin(theta2)
    xr2 =  c2*(x - cx) + s2*(y - cy)
    yr2 = -s2*(x - cx) + c2*(y - cy)
    r2 = np.mean(((xr2 / b)**2 + (yr2 / a)**2 - 1.0)**2)  # note: a<->b

    if r2 < r1:
        a, b, theta = b, a, theta2
    # Final normalize to keep θ in canonical range and a ≥ b
    a, b, theta = _normalize_axes_angle(a, b, theta)
    return a, b, theta


def _ellipse_residuals(cx, cy, a, b, theta, x, y):
    """
    Algebraic residuals |(xr/a)^2 + (yr/b)^2 - 1| for points (x,y).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * (x - cx) + s * (y - cy)
    yr = -s * (x - cx) + c * (y - cy)
    return np.abs((xr / a)**2 + (yr / b)**2 - 1.0)





# --- Alignment helpers (image coords: y goes down) -----------------------

def _rotate_points_image(x, y, cx, cy, ang):
    """
    Rotate points (x,y) around center (cx,cy) by angle ang (radians)
    in IMAGE coordinates (x right, y down).
    Convention: positive ang rotates clockwise visually (since y down).
    We implement by flipping to math, rotating CCW, flip back.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    # image -> math (flip y)
    ym = -(y - cy)
    xm =  (x - cx)
    c, s = np.cos(ang), np.sin(ang)
    # rotate in math (+ang CCW)
    xr = c * xm - s * ym
    yr = s * xm + c * ym
    # back to image
    X = cx + xr
    Y = cy - yr
    return X, Y

def align_to_x_axis_image(x, y, cx, cy, a, b, theta):
    """
    Rotate everything so that the ellipse’s MAJOR axis becomes horizontal.
    Returns: (x_rot, y_rot, cx, cy, a_aligned, b_aligned, theta_aligned=0)
    NOTE: we keep center fixed, we only rotate around (cx,cy).
    """
    # On veut major axis horizontal => angle affiché = 0.
    # En image coords, pour annuler theta on applique une rotation de -theta.
    x_rot, y_rot = _rotate_points_image(x, y, cx, cy, -theta)
    # Une fois aligné, l’ellipse a le même (a,b), mais angle 0
    return x_rot, y_rot, cx, cy, a, b, 0.0






# --- Direct Least-Squares Ellipse Fit ------------------------------------

def fit_ellipse_ls(x: np.ndarray, y: np.ndarray, ridge: float = 1e-8):
    """
    Direct least-squares fit of an ellipse (Fitzgibbon 1999).
    Returns (cx, cy, a, b, theta) with a ≥ b and θ ∈ [-π/2, π/2).

    Notes
    -----
    - Hartley-style normalization (mean-center + RMS scale) improves conditioning.
    - A small ridge is added to S22; we fall back to pinv if needed.
    - Orientation ambiguity (±90°) is resolved by _best_orientation().
    """
    if len(x) < 6:
        raise ValueError("Need ≥ 6 points for ellipse fit.")
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)

    # Normalize (mean-center + RMS scale)
    xm, ym = float(x.mean()), float(y.mean())
    X0, Y0 = x - xm, y - ym
    s = float(np.sqrt(np.mean(X0**2 + Y0**2))) or 1.0
    X, Y = X0 / s, Y0 / s

    # Design matrix D = [x^2, xy, y^2, x, y, 1]
    D = np.hstack([X*X, X*Y, Y*Y, X, Y, np.ones_like(X)])
    S = D.T @ D
    S11, S12 = S[:3, :3], S[:3, 3:]
    S21, S22 = S[3:, :3], S[3:, 3:]

    # Solve T = -inv(S22) @ S21 with ridge + pinv fallback
    lam = ridge * (np.trace(S22) / 3.0 + 1.0)
    S22r = S22 + lam * np.eye(3)
    try:
        T = -np.linalg.solve(S22r, S21)
    except np.linalg.LinAlgError:
        T = -np.linalg.pinv(S22r) @ S21

    # Reduced 3x3 problem
    M = S11 + S12 @ T

    # Ellipse constraint and generalized eigen
    C = np.array([[0.0, 0.0, 2.0],
                  [0.0,-1.0, 0.0],
                  [2.0, 0.0, 0.0]])
    Cinvm = np.linalg.pinv(C) @ M
    _, V = np.linalg.eig(Cinvm)

    # Pick eigenvector satisfying 4ac - b^2 > 0
    v = None
    for k in range(3):
        a_, b_, c_ = V[:, k].real
        if 4.0 * a_ * c_ - b_ * b_ > 0.0:
            v = V[:, k].real
            break
    if v is None:
        raise RuntimeError("No valid ellipse eigenvector (constraint failed).")

    a_, b_, c_ = v
    d_, e_, f_ = (T @ v).ravel()

    den = b_*b_ - 4.0*a_*c_
    if abs(den) < 1e-14:
        raise RuntimeError("Degenerate conic (den≈0).")

    # Center (normalized coords)
    cx_n = (2.0*c_*d_ - b_*e_) / den
    cy_n = (2.0*a_*e_ - b_*d_) / den

    # Angle
    theta = 0.5 * np.arctan2(b_, (a_ - c_))

    # Evaluate conic at the center -> F0
    F0 = a_*cx_n*cx_n + b_*cx_n*cy_n + c_*cy_n*cy_n + d_*cx_n + e_*cy_n + f_
    A = np.array([[a_, b_/2.0], [b_/2.0, c_]])
    w, _ = np.linalg.eigh(A)  # ascending eigenvalues
    a_len_n = np.sqrt(abs(-F0 / w[0]))
    b_len_n = np.sqrt(abs(-F0 / w[1]))

    # Un-normalize back to original coords
    cx = xm + s * cx_n
    cy = ym + s * cy_n
    a_len = s * a_len_n
    b_len = s * b_len_n
    # Canonical orientation + residual-based disambiguation (θ vs θ+π/2)
    a_len, b_len, theta = _normalize_axes_angle(a_len, b_len, theta)
    a_len, b_len, theta = _best_orientation(x.ravel(), y.ravel(), cx, cy, a_len, b_len, theta)
    # NEW: PCA-based alignment to match the cloud's principal direction
    a_len, b_len, theta = _align_with_pca(a_len, b_len, theta, x.ravel(), y.ravel())

    return float(cx), float(cy), float(a_len), float(b_len), float(theta)


# --- Angle helpers / PCA alignment ---------------------------------------

def _angle_wrap_pi(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def _angle_diff_mod_pi(a: float, b: float) -> float:
    """Smallest absolute difference between angles a and b, modulo pi."""
    # (ellipse axes are unoriented lines → modulo π, not 2π)
    d = _angle_wrap_pi(a - b)
    # normalize to [0, pi)
    if d < 0:
        d = -d
    if d >= np.pi / 2.0:
        d = np.pi - d
    return d

def _pca_angle(x: np.ndarray, y: np.ndarray) -> float:
    """
    Principal direction angle (in image coords: x right, y down).
    Returns angle in radians w.r.t. +x axis.
    """
    X = np.asarray(x, float).ravel()
    Y = np.asarray(y, float).ravel()
    Xc = X - X.mean()
    Yc = Y - Y.mean()
    C = np.cov(np.vstack((Xc, Yc)))  # 2x2
    w, V = np.linalg.eigh(C)         # ascending
    v = V[:, 1]                      # principal eigenvector
    angle = np.arctan2(v[1], v[0])   # image coords
    # canonical wrap to [-pi/2, pi/2)
    angle = (angle + np.pi/2.0) % np.pi - np.pi/2.0
    return angle

def _align_with_pca(a: float, b: float, theta: float,
                    x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Choose between (a,b,theta) and (b,a,theta+pi/2) so that the major axis
    aligns with the point cloud's principal direction (PCA).
    """
    # compute principal direction of the cloud
    th_pca = _pca_angle(x, y)
    # candidate 1
    d1 = _angle_diff_mod_pi(theta, th_pca)
    # candidate 2 (swap axes + rotate)
    theta2 = (theta + np.pi/2.0)
    d2 = _angle_diff_mod_pi(theta2, th_pca)
    if d2 < d1:
        a, b, theta = b, a, theta2
    # final canonicalization (a≥b, theta in [-pi/2, pi/2))
    a, b, theta = _normalize_axes_angle(a, b, theta)
    return a, b, theta

def convert_angle(theta: float, from_convention: str, to_convention: str) -> float:
    """
    Convert angle between 'image' (x right, y down) and 'math' (x right, y up).
    For lines/ellipse axes we work modulo π, so sign flips matter.
    """
    if from_convention == to_convention:
        return theta
    if from_convention == "image" and to_convention == "math":
        return -theta
    if from_convention == "math" and to_convention == "image":
        return -theta
    raise ValueError("Unknown convention; use 'image' or 'math'.")


# --- RANSAC wrapper -------------------------------------------------------

def fit_ellipse_ransac(x: np.ndarray, y: np.ndarray,
                       iters: int = 800,
                       sample_size: int = 7,
                       inlier_thresh: float = 0.25,
                       min_inliers: int = 20):
    """
    Robust ellipse fitting using RANSAC on top of fit_ellipse_ls():
      - Randomly sample 'sample_size' points, fit, measure inliers by residual threshold.
      - Keep the best model by inlier count and refit on its inliers.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinates of candidate boundary points.
    iters : int
        Number of random hypotheses to try.
    sample_size : int
        Points per hypothesis (≥ 6 recommended).
    inlier_thresh : float
        Residual threshold for inlier classification (algebraic residual).
    min_inliers : int
        Minimum consensus to accept a model; fallback to LS if not reached.

    Returns
    -------
    (cx, cy, a, b, theta)
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = len(x)
    if n < 6:
        raise ValueError("Need ≥ 6 points for ellipse fit.")
    rng = np.random.default_rng(42)

    best = None
    for _ in range(max(1, iters)):
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        try:
            cx, cy, a, b, th = fit_ellipse_ls(x[idx], y[idx])
            res = _ellipse_residuals(cx, cy, a, b, th, x, y)
            inliers = res < inlier_thresh
            score = int(inliers.sum())
            if best is None or score > best[0]:
                best = (score, inliers, (cx, cy, a, b, th))
        except Exception:
            continue

    if best is None or best[0] < max(min_inliers, 6):
        # Fallback to plain LS if RANSAC didn't find a decent consensus
        return fit_ellipse_ls(x, y)

    # Refit on inliers
    mask = best[1]
    return fit_ellipse_ls(x[mask], y[mask])


# --- Sampling -------------------------------------------------------------

def ellipse_points(cx: float, cy: float, a: float, b: float, theta: float, n: int = 400):
    """
    Generate n sampled points on the ellipse (no plotting).
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    X = a * np.cos(t)
    Y = b * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * X - s * Y
    yr = s * X + c * Y
    return cx + xr, cy + yr



# --- Image-convention sampling -------------------------------------------

# --- Sampling (image coordinates: y increases downward) -------------------

def ellipse_points_image(cx: float, cy: float, a: float, b: float, theta: float, n: int = 400):
    """
    Sample points on the ellipse but return them in IMAGE coordinates
    (x right, y down) so they overlay correctly on imshow(origin='upper').

    This is equivalent to sampling in math coords and then flipping y.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    X = a * np.cos(t)
    Y = b * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)

    # rotate in the ellipse frame (math), then convert to image coords by flipping y
    xr = c * X - s * Y
    yr = s * X + c * Y

    # map to pixels: image-y goes downward -> flip sign on yr
    return cx + xr, cy - yr

# --- Geometric refinement (Levenberg–Marquardt, NumPy-only) ---------------

def _residuals_geom(x, y, cx, cy, a, b, th):
    c, s = np.cos(th), np.sin(th)
    xr = c * (x - cx) + s * (y - cy)
    yr = -s * (x - cx) + c * (y - cy)
    return (xr / a) ** 2 + (yr / b) ** 2 - 1.0

def _jacobian_num(x, y, cx, cy, a, b, th, eps=1e-6):
    """
    Numerical Jacobian of residuals wrt params [cx, cy, a, b, th].
    Shape: (N, 5)
    """
    r0 = _residuals_geom(x, y, cx, cy, a, b, th)
    J = np.empty((x.size, 5), dtype=float)

    def col(pname, base, step):
        args = [cx, cy, a, b, th]
        i = ["cx", "cy", "a", "b", "th"].index(pname)
        args[i] = base + step
        return (_residuals_geom(x, y, *args) - r0) / step

    J[:, 0] = col("cx", cx, eps)
    J[:, 1] = col("cy", cy, eps)
    J[:, 2] = col("a",  a,  eps)
    J[:, 3] = col("b",  b,  eps)
    J[:, 4] = col("th", th, eps)
    return J

def refine_ellipse_lm(x, y, cx, cy, a, b, th,
                      iters: int = 25, lam: float = 1e-2, tol: float = 1e-9):
    """
    Levenberg–Marquardt refinement of ellipse params in image coords.
    Minimizes mean squared algebraic residuals of the implicit ellipse.
    Returns improved (cx, cy, a, b, th).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    cx, cy, a, b, th = float(cx), float(cy), float(a), float(b), float(th)

    # keep major axis first during refinement
    if b > a:
        a, b, th = b, a, th + np.pi/2.0

    best = (np.inf, (cx, cy, a, b, th))
    for _ in range(max(1, iters)):
        r = _residuals_geom(x, y, cx, cy, a, b, th)
        cost = float(np.mean(r * r))
        if cost < best[0]:
            best = (cost, (cx, cy, a, b, th))
        # Jacobian (N x 5)
        J = _jacobian_num(x, y, cx, cy, a, b, th)
        # Normal equations with damping
        H = J.T @ J
        g = J.T @ r
        H_lm = H + lam * np.eye(5)
        try:
            delta = -np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(H_lm) @ g

        cx_new = cx + delta[0]
        cy_new = cy + delta[1]
        a_new  = max(1e-6, a + delta[2])
        b_new  = max(1e-6, b + delta[3])
        th_new = th + delta[4]

        # evaluate new cost
        r_new = _residuals_geom(x, y, cx_new, cy_new, a_new, b_new, th_new)
        cost_new = float(np.mean(r_new * r_new))

        if cost_new < cost:
            # accept step, slightly reduce damping
            cx, cy, a, b, th = cx_new, cy_new, a_new, b_new, th_new
            lam = max(1e-6, lam * 0.7)
            if abs(cost - cost_new) < tol:
                break
        else:
            # reject step, increase damping
            lam *= 2.5

    cx, cy, a, b, th = best[1]
    # final canonicalization + best θ vs θ+π/2 + PCA align like before
    a, b, th = _normalize_axes_angle(a, b, th)
    a, b, th = _best_orientation(x, y, cx, cy, a, b, th)
    a, b, th = _align_with_pca(a, b, th, x, y)
    return float(cx), float(cy), float(a), float(b), float(th)
