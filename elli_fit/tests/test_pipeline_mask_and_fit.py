"""
===========================================================
Pipeline Tests (edge_points -> fit_ellipse_ls)
===========================================================
"""

import numpy as np
from elli_fit.core import edge_points, fit_ellipse_ls


def test_pipeline_simple_mask():
    # Build a simple filled ellipse mask in a grid
    cx, cy, a, b, th = 20.0, 18.0, 10.0, 6.0, np.deg2rad(15)
    H = W = 48
    Y, X = np.mgrid[0:H, 0:W]
    xr =  (X - cx)*np.cos(th) + (Y - cy)*np.sin(th)
    yr = -(X - cx)*np.sin(th) + (Y - cy)*np.cos(th)
    M = ((xr/a)**2 + (yr/b)**2 <= 1.0).astype(float)

    # Detect edges and fit
    x, y = edge_points(M, threshold=0.5)
    cx2, cy2, a2, b2, th2 = fit_ellipse_ls(x, y)

    # sanity checks
    assert 0 <= cx2 <= W and 0 <= cy2 <= H
    assert a2 > 0 and b2 > 0
    assert a2 >= b2
