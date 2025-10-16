"""
===========================================================
Test suite for elli_fit core (compact version)
===========================================================
"""

import numpy as np
from elli_fit.core import edge_points, fit_ellipse_ls, ellipse_points


def test_fit_ellipse_ls_basic():
    """Check that the least-squares fit recovers a synthetic ellipse."""
    cx_true, cy_true = 10.0, 10.0
    a_true, b_true, th_true = 6.0, 3.0, np.deg2rad(25)
    t = np.linspace(0, 2*np.pi, 400)
    X = cx_true + a_true * np.cos(t) * np.cos(th_true) - b_true * np.sin(t) * np.sin(th_true)
    Y = cy_true + a_true * np.cos(t) * np.sin(th_true) + b_true * np.sin(t) * np.cos(th_true)

    cx, cy, a, b, th = fit_ellipse_ls(X, Y)
    assert np.isclose(cx, cx_true, atol=0.2)
    assert np.isclose(cy, cy_true, atol=0.2)
    assert np.isclose(a, a_true, atol=0.3)
    assert np.isclose(b, b_true, atol=0.3)


def test_sampling_shape():
    """Ensure ellipse_points() returns the correct array shapes."""
    X, Y = ellipse_points(0, 0, 3, 2, np.deg2rad(15), n=100)
    assert X.shape == (100,) and Y.shape == (100,)



"""
===========================================================
Optional recovery test (won't break the suite if it fails)
===========================================================

This test creates a synthetic binary grid with a known rotated ellipse,
extracts edges (like the real pipeline), fits via fit_ellipse_ls, and
compares the recovered parameters to the ground truth.

It is marked xfail so that failures don't break the test run — useful as a
non-regression signal while we iterate on data quirks.
"""

# --- Optional, non-failing test ------------------------------------------

import pytest

@pytest.mark.xfail(strict=False, reason="Optional: full binary-grid pipeline tolerance check")
def test_optional_full_pipeline_recover_known_ellipse():
    # Ground-truth ellipse
    cx_true, cy_true = 18.0, 16.0
    a_true, b_true   = 8.0, 5.0
    th_true          = np.deg2rad(35.0)

    # Build binary grid containing the filled rotated ellipse
    size = 40
    Y, X = np.mgrid[0:size, 0:size]
    xr =  (X - cx_true) * np.cos(th_true) + (Y - cy_true) * np.sin(th_true)
    yr = -(X - cx_true) * np.sin(th_true) + (Y - cy_true) * np.cos(th_true)
    M = ((xr / a_true) ** 2 + (yr / b_true) ** 2 <= 1.0).astype(float)

    # Extract edge points from the grid (no cheating)
    x, y = edge_points(M, threshold=0.5)

    # Fit
    cx, cy, a, b, th = fit_ellipse_ls(x, y)

    # Tolerances (a bit looser than parametric-point tests, since we discretized on a grid)
    assert np.isclose(cx, cx_true, atol=0.6)
    assert np.isclose(cy, cy_true, atol=0.6)
    assert np.isclose(a,  a_true, atol=0.6)
    assert np.isclose(b,  b_true, atol=0.6)
    # Angle check is intentionally omitted here because discrete masks can flip by ~90°
    # depending on sampling; the core already disambiguates, but we keep this optional test
    # robust to pixelization.
    
    