"""
===========================================================
Examples (CLI) Smoke Test
===========================================================

Smoke-test the CLI example to make sure it runs and writes CSVs.
"""

import importlib.util
from pathlib import Path
import numpy as np


# --- Helper ---------------------------------------------------------------

def _write_mask_csv(path: Path, M: np.ndarray):
    np.savetxt(path, M, delimiter=",", fmt="%.6f")


def _import_demo_cli():
    """
    Dynamically import examples/demo_cli.py using its absolute path.
    Works even when pytest runs in a tmpdir.
    """
    project_root = Path(__file__).resolve().parents[1]
    demo_path = project_root / "examples" / "demo_cli.py"
    spec = importlib.util.spec_from_file_location("demo_cli", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



# --- Test ----------------------------------------------------------------

def test_demo_cli_exports(tmp_path: Path):
    # Prepare grid
    cx, cy, a, b, th = 16.0, 16.0, 7.0, 4.0, np.deg2rad(20)
    H = W = 32
    Y, X = np.mgrid[0:H, 0:W]
    xr =  (X - cx)*np.cos(th) + (Y - cy)*np.sin(th)
    yr = -(X - cx)*np.sin(th) + (Y - cy)*np.cos(th)
    M = ((xr/a)**2 + (yr/b)**2 <= 1.0).astype(float)

    csv_path = tmp_path / "grid.csv"
    _write_mask_csv(csv_path, M)

    # Import demo_cli safely
    demo = _import_demo_cli()

    # Run main
    exit_code = demo.main([str(csv_path)])
    assert exit_code == 0

    # Check exports in current working directory of the test
    edge_csv = Path("edge_points.csv")
    fit_csv  = Path("fitted_ellipse.csv")
    assert edge_csv.exists(), "edge_points.csv not created"
    assert fit_csv.exists(), "fitted_ellipse.csv not created"

    # Cleanup
    edge_csv.unlink(missing_ok=True)
    fit_csv.unlink(missing_ok=True)
