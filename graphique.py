from main import *
# Diamètres à tracer (mm)
D_values = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

# Calcul des paramètres de forme
alpha_vals = alpha_N(D_values)
a_vals, b_vals = semi_axes_from_D_alpha(D_values, alpha_vals)

# Tracé des profils méridiens
theta = np.linspace(0, 2*np.pi, 400)
fig2, ax2 = plt.subplots(figsize=(6,6))
for D, a, b, al in zip(D_values, a_vals, b_vals, alpha_vals):
    x = a * np.cos(theta)
    z = b * np.sin(theta)
    ax2.plot(x, z, label=f"D={D:.1f} mm, α={al:.2f}")
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("z (mm)")
ax2.set_title("Profils méridiens (sphéroïde oblat) pour différents D")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tableau récap
df = pd.DataFrame({
    "D_mm": D_values,
    "alpha_b_over_a": alpha_vals,
    "a_semi_axis_mm": a_vals,
    "b_semi_axis_mm": b_vals,
})
print(df)

# (Option) Tracé style Fig. 6 de l'article : 
# Ici on ne conserve pas le volume réel, on veut comparer les FORMES.
# On fixe donc a = 1 (largeur normalisée), et on ajuste b = α * a pour chaque diamètre.
# Cela permet de voir uniquement l'aplatissement relatif en fonction de D.

alpha_vals = alpha_N(D_values)   # On calcule α = b/a pour chaque diamètre
a_norm = 1.0                     # On fixe le demi-grand axe horizontal à 1 (normalisation)
theta = np.linspace(0, 2*np.pi, 600)  # Paramètre angulaire pour tracer l'ellipse

fig, ax = plt.subplots(figsize=(6,6))
for D, al in zip(D_values, alpha_vals):
    # Equation d'une ellipse normalisée : x = a cosθ, z = b sinθ
    x = a_norm * np.cos(theta)
    z = (al * a_norm) * np.sin(theta)   # b = α * a (avec a=1 donc b=α)
    
    # On trace la forme pour chaque diamètre
    ax.plot(x, z, label=f"D={D:.1f} mm, α={al:.2f}")

# Mise en forme du graphe
ax.set_aspect("equal")                   # Echelle identique en x et z
ax.set_xlabel("x / a (normalisé)")       # Axe horizontal normalisé
ax.set_ylabel("z / a (normalisé)")       # Axe vertical normalisé
ax.set_title("Formes normalisées (a = 1) — style Fig. 6")
ax.legend(fontsize=8, loc="upper right") # Légende avec D et α
ax.grid(True, alpha=0.3)                 # Grille légère
plt.tight_layout()
plt.show()
