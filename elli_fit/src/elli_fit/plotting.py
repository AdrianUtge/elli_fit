import matplotlib.pyplot as plt
import numpy as np

def plot_fit(x, y, X_fit, Y_fit, a, b, title="Ellipse ajustée"):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=20, label="pixels = 1 (nuage)")
    plt.plot(X_fit, Y_fit, linewidth=2, label=f"ellipse fit (a≈{a:.2f}, b≈{b:.2f})")
    ax = plt.gca()
    ax.set_aspect("equal", "box")
    plt.legend()
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()
