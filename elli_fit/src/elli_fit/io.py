from pathlib import Path
import numpy as np


def load_binary_matrix(path: str, delimiter: str = ",", skiprows: int = 0) -> np.ndarray:
    """
    Load a numeric 2D matrix (CSV).

    Parameters
    ----------
    path : str
        CSV file path.
    delimiter : str
        CSV delimiter (default ",").
    skiprows : int
        Number of initial rows to skip (useful if the CSV has a header line).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return np.loadtxt(p, delimiter=delimiter, skiprows=skiprows)


def save_xy_csv(path: str, x, y):
    """
    Save paired (x, y) coordinates to a CSV.

    Parameters
    ----------
    path : str
        Output CSV file path.
    x, y : array-like
        Sequences of equal length containing coordinates.

    Notes
    -----
    This is useful to inspect results or to plot them with external tools.
    """
    arr = np.column_stack([x, y])
    np.savetxt(path, arr, delimiter=",", header="x,y", comments="", fmt="%.6f")
