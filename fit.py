import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Charger la matrice binaire (20x20 générée précédemment)
M = pd.read_csv("/Users/adrianutge/Library/Mobile Documents/iCloud~md~obsidian/Documents/Vault_AD/FAC/Stage/Code/Résumé ellipsoïde 20x20.csv").values

# Extraire les points où M=1
i, j = np.where(M > 0.5)
x = j + 0.5
y = i + 0.5

# Recentrer (barycentre comme estimation du centre)
cx, cy = np.mean(x), np.mean(y)
x_c = x - cx
y_c = y - cy

# Fonction coût
def cost(params):
    a, b = params
    if a <= 0 or b <= 0:
        return 1e9
    val = (x_c**2 / a**2) + (y_c**2 / b**2)
    return np.mean((val - 1)**2)

# Fit
res = minimize(cost, x0=[7, 5])
a_fit, b_fit = res.x

# Points de l'ellipse fit
theta = np.linspace(0, 2*np.pi, 400)
X_fit = cx + a_fit * np.cos(theta)
Y_fit = cy + b_fit * np.sin(theta)

# Plot
plt.figure(figsize=(5,5))
plt.scatter(x, y, s=20, label="pixels=1 (nuage)", c="blue")
plt.plot(X_fit, Y_fit, 'r', lw=2, label=f"ellipse fit (a≈{a_fit:.2f}, b≈{b_fit:.2f})")
plt.gca().set_aspect("equal", "box")
plt.legend()
plt.title("Ellipse fit superposée au nuage de points")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.show()
