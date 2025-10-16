"""
===========================================================
Ellipse Fitting Demo (CLI version, NumPy-only)
===========================================================

Usage
-----
    python3 examples/demo_cli.py path/to/grid.csv

Outputs
-------
    edge_points.csv
    fitted_ellipse.csv
"""

# --- Imports --------------------------------------------------------------

import sys
from pathlib import Path
import numpy as np
from elli_fit import load_binary_matrix, save_xy_csv, edge_points, fit_ellipse_ls, ellipse_points


# --- Main routine ---------------------------------------------------------

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("Usage: demo_cli.py path/to/file.csv")
        return 1

    csv_path = Path(argv[0])
    M = load_binary_matrix(csv_path)
    x, y = edge_points(M, threshold=0.5)
    cx, cy, a, b, th = fit_ellipse_ls(x, y)
    Xf, Yf = ellipse_points(cx, cy, a, b, th)

    save_xy_csv("edge_points.csv", x, y)
    save_xy_csv("fitted_ellipse.csv", Xf, Yf)

    print(f"center=({cx:.2f},{cy:.2f}), a={a:.2f}, b={b:.2f}, θ={np.degrees(th):.2f}°")
    print("✅ Exported 'edge_points.csv' and 'fitted_ellipse.csv'")
    return 0


# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
