# Documentation du dossier `src/elli_fit`

## Présentation générale

Le dossier `elli_fit` contient une bibliothèque légère pour la détection et l’ajustement d’ellipses à partir de grilles 2D binaires ou en niveaux de gris (par exemple, des images au format CSV). Elle est conçue pour être minimaliste, robuste et ne dépend que de NumPy.

---

## Fichiers principaux

### 1. `core.py`

Ce fichier implémente le cœur algorithmique :
- **Prétraitement des masques** (`preprocess_mask`) : binarisation et nettoyage des artefacts simples sur une image 2D.
- **Extraction des contours** (`edge_points`, `edge_points_ordered`) : obtention des coordonnées des pixels de bord, ordonnés ou non.
- **Ajustement d’ellipse** (`fit_ellipse_ls`, `fit_ellipse_ransac`) : ajustement direct par moindres carrés (méthode de Fitzgibbon) ou version robuste via RANSAC.
- **Génération de points sur une ellipse** (`ellipse_points`) : échantillonnage de points sur une ellipse ajustée.

Le code vise la clarté, la robustesse et la reproductibilité, avec des normalisations et des sécurités numériques.

---

### 2. `io.py`

Ce module gère les entrées/sorties :
- **Chargement de matrices** (`load_binary_matrix`) : lit une matrice 2D depuis un fichier CSV, avec gestion du séparateur et des lignes d’en-tête.
- **Sauvegarde de coordonnées** (`save_xy_csv`) : exporte des paires de coordonnées (x, y) dans un CSV, utile pour l’inspection ou la visualisation.

---

### 3. `__init__.py`

Ce fichier expose les fonctions principales de la bibliothèque et donne un exemple de workflow typique :
- Chargement d’une matrice : `M = load_binary_matrix("grid.csv")`
- Extraction des points de bord : `x, y = edge_points(M)`
- Ajustement d’une ellipse : `params = fit_ellipse_ls(x, y)`
- Génération de points sur l’ellipse : `xp, yp = ellipse_points(*params)`

---

## Exemple d’utilisation

```python
from elli_fit import *
M = load_binary_matrix("grid.csv")
x, y = edge_points(M)
params = fit_ellipse_ls(x, y)
xp, yp = ellipse_points(*params)
save_xy_csv("ellipse_fit.csv", xp, yp)
```

---

## Auteur

Adrian Utge Le Gall, 2025

---

## Détail mathématique et fonctionnement des fonctions principales

### 1. `preprocess_mask`
- **But** : Binariser une image (seuil) puis nettoyer les artefacts (pixels isolés, bords).
- **Maths** :
    - Seuil : $B = (M > threshold)$
    - Nettoyage : on garde les pixels ayant au moins `min_neighbors` voisins dans leur voisinage 3x3.
    - Optionnel : on met à zéro les bords (utile pour les images CSV avec artefacts sur les bords).

### 2. `edge_points`
- **But** : Extraire les coordonnées (x, y) des centres des pixels de bord (non ordonnés).
- **Maths** :
    - Un pixel de bord est un pixel foreground (1) qui a au moins un voisin (haut, bas, gauche, droite) à 0.
    - Retourne les centres des pixels de bord.

### 3. `edge_points_ordered`
- **But** : Extraire un contour ordonné (tracé continu) des bords.
- **Maths** :
    - Algorithme de Moore-Neighbor tracing (8-voisinage, sens horaire).
    - Retourne les coordonnées ordonnées du contour principal trouvé.

### 4. `fit_ellipse_ls`
- **But** : Ajuster une ellipse par moindres carrés directs (méthode de Fitzgibbon 1999).
- **Maths** :
    - On cherche les paramètres $(a, b, \theta, c_x, c_y)$ d’une ellipse qui minimisent l’erreur algébrique sur les points donnés.
    - Utilise la normalisation de Hartley (centrage, mise à l’échelle) pour la stabilité numérique.
    - Résout un problème d’autovecteurs généralisés sous contrainte d’ellipse ($4ac-b^2>0$).
    - Désambiguïse l’orientation (axe majeur, angle dans $[-\pi/2, \pi/2)$) et aligne avec la direction principale (PCA).

### 5. `fit_ellipse_ransac`
- **But** : Ajustement robuste d’ellipse par RANSAC.
- **Maths** :
    - Tire aléatoirement des sous-ensembles de points, ajuste une ellipse, compte les inliers (points proches de l’ellipse).
    - Garde le modèle avec le plus d’inliers, puis réajuste sur ces inliers.
    - Permet de résister aux points aberrants.

### 6. `ellipse_points`
- **But** : Générer des points échantillonnés sur une ellipse donnée.
- **Maths** :
    - Paramétrisation : $x = c_x + a \cos t \cos\theta - b \sin t \sin\theta$
    - $y = c_y + a \cos t \sin\theta + b \sin t \cos\theta$
    - $t$ varie de $0$ à $2\pi$ (n points).

### 7. `refine_ellipse_lm`
- **But** : Raffiner les paramètres d’une ellipse par Levenberg–Marquardt (optimisation non linéaire).
- **Maths** :
    - Minimise la somme des carrés des résidus géométriques $(xr/a)^2 + (yr/b)^2 - 1$.
    - Utilise le calcul du Jacobien numérique et une descente de gradient avec amortissement.

---

## Résumé mathématique

- **Extraction de contours** : détection de pixels de bord par voisinage.
- **Ajustement d’ellipse** : résolution d’un problème quadratique sous contrainte, puis désambiguïsation de l’orientation.
- **RANSAC** : sélection robuste en présence de bruit ou d’outliers.
- **Optimisation non linéaire** : raffinement local des paramètres pour minimiser l’erreur géométrique réelle.




python3 case_shadow_triptych.py \
  "test_data/Capture Oct 20 2025.png" \
  --mm-per-px 0.045 \
  --refine \
  --ransac-iters 1500 \
  --save-dir out_triptych_debug \
  --debug-candidates
