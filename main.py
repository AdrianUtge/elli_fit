import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def alpha_N(D_mm: np.ndarray) -> np.ndarray:
    """
    α = b/a (rapport d’axes).
    ⚠️ Le polynôme est défini pour D en centimètres (cm), pas en mm.
    On convertit donc D_mm -> D_cm avant d’évaluer.
    Plage de validité typique ~ 0.5 mm à 6 mm.
    """
    D_cm = D_mm / 10.0
    alpha = 1.0048 + 0.0057*D_cm - 2.628*(D_cm**2) + 3.682*(D_cm**3) - 1.677*(D_cm**4)
    # Sécurité numérique / physique : on borne α dans [0.3, 1.05]
    alpha = np.clip(alpha, 0.3, 1.05)
    return alpha

def semi_axes_from_D_alpha(D_mm: np.ndarray, alpha: np.ndarray):
    """
    Volume conservé entre sphère équivalente (diamètre D) et sphéroïde oblat (a,a,b).
    V_sphère = (π/6) D^3  et  V_sphéroïde = (4/3) π a^2 b
    => a^3 = D^3 / (8 α)  puis  b = α a
    """
    # On évite toute division par ~0 au cas où
    alpha_safe = np.maximum(alpha, 1e-6)
    a = (D_mm**3 / (8.0 * alpha_safe)) ** (1.0/3.0)  # mm
    b = alpha_safe * a                                # mm
    return a, b

