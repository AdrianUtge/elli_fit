"""
===========================================================
elli_fit — lightweight ellipse fitting library
===========================================================

A minimal, self-contained NumPy-based toolkit for detecting and fitting
ellipses from binary or grayscale 2D grids (e.g., CSV images).

Main functions
--------------
- load_binary_matrix(path)
- save_xy_csv(path, x, y)
- edge_points(M, threshold=0.5)
- fit_ellipse_ls(x, y)
- ellipse_points(cx, cy, a, b, theta)

Typical workflow
----------------
    from elli_fit import *
    M = load_binary_matrix("grid.csv")
    x, y = edge_points(M)
    cx, cy, a, b, th = fit_ellipse_ls(x, y)
    Xf, Yf = ellipse_points(cx, cy, a, b, th)

Author
------
Adrian Utge Le Gall, 2025
"""

# --- Public Imports -------------------------------------------------------

from .io import load_binary_matrix, save_xy_csv
from .core import edge_points, fit_ellipse_ls, ellipse_points, refine_ellipse_lm

__all__ = [
    "load_binary_matrix",
    "save_xy_csv",
    "edge_points",
    "fit_ellipse_ls",
    "ellipse_points",
    "refine_ellipse_lm",
]
