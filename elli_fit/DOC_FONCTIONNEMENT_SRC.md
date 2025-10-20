# Documentation de la bibliothèque `elli_fit`

## 1. Présentation générale

`elli_fit` est une bibliothèque Python minimaliste conçue pour la détection et l'ajustement d'ellipses à partir de données 2D (images, masques binaires). Son objectif est de fournir des outils robustes et performants avec un minimum de dépendances (principalement NumPy).

Elle est particulièrement adaptée aux applications de vision par ordinateur où il est nécessaire d'analyser des formes circulaires ou elliptiques, comme la mesure d'objets à partir de leurs ombres projetées.

---

## 2. Architecture de la bibliothèque (`src/elli_fit`)

Le cœur de la bibliothèque se trouve dans le dossier `src/elli_fit` et s'articule autour de deux modules principaux.

### `core.py` — Le cœur algorithmique

Ce fichier contient toutes les fonctions mathématiques et algorithmiques pour le traitement des données et l'ajustement des ellipses.

- **Prétraitement** :
    - `preprocess_mask`: Binarise et nettoie une image 2D pour isoler les formes d'intérêt.
- **Extraction de contours** :
    - `edge_points`: Extrait les coordonnées des pixels de contour (non ordonnés).
    - `edge_points_ordered`: Extrait un contour unique et ordonné en utilisant un algorithme de suivi de contour (Moore-Neighbor).
- **Ajustement d'ellipse** :
    - `fit_ellipse_ls`: Ajuste une ellipse par la méthode des moindres carrés directs (Fitzgibbon). Rapide mais sensible au bruit.
    - `fit_ellipse_ransac`: Implémente l'algorithme RANSAC pour un ajustement robuste en présence de points aberrants (outliers).
    - `refine_ellipse_lm`: Raffine les paramètres d'une ellipse en utilisant l'algorithme de Levenberg-Marquardt pour minimiser l'erreur géométrique.
- **Utilitaires** :
    - `ellipse_points` / `ellipse_points_image`: Génère des points sur le périmètre d'une ellipse définie par ses paramètres.

### `io.py` — Gestion des entrées/sorties

Ce module simplifie la lecture et l'écriture de données.

- `load_binary_matrix`: Charge une matrice 2D depuis un fichier texte (type CSV).
- `save_xy_csv`: Sauvegarde des coordonnées (x, y) dans un fichier CSV, utile pour la visualisation ou le débogage.

### `__init__.py`

Ce fichier expose les fonctions publiques de la bibliothèque, permettant de les importer directement depuis le package `elli_fit`. Il définit l'API principale de la bibliothèque.

---

## 3. Principes mathématiques et algorithmes

### Ajustement par moindres carrés (`fit_ellipse_ls`)

La méthode s'appuie sur les travaux de Fitzgibbon & Fisher (1999). Elle consiste à trouver les coefficients de l'équation générale d'une conique ($Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0$) qui minimisent l'erreur algébrique pour un ensemble de points. La contrainte $B^2 - 4AC < 0$ est appliquée pour garantir que la conique est une ellipse. Ce problème est résolu efficacement via une décomposition en valeurs propres.

### Robustesse par RANSAC (`fit_ellipse_ransac`)

Pour les données bruitées, RANSAC (RANdom SAmple Consensus) est une approche itérative :
1.  Sélectionner un sous-ensemble aléatoire de points (suffisant pour définir une ellipse).
2.  Ajuster une ellipse sur ce sous-ensemble avec `fit_ellipse_ls`.
3.  Compter le nombre de points "inliers" : ceux qui sont suffisamment proches de l'ellipse ajustée.
4.  Répéter N fois et conserver l'ellipse qui maximise le nombre d'inliers.
5.  Ré-ajuster l'ellipse finale en utilisant tous les inliers identifiés.

### Raffinement non linéaire (`refine_ellipse_lm`)

L'ajustement par moindres carrés minimise une erreur *algébrique*, qui n'est pas toujours représentative de la distance géométrique réelle. L'algorithme de Levenberg-Marquardt est une méthode d'optimisation non linéaire qui affine les paramètres de l'ellipse (`cx`, `cy`, `a`, `b`, `theta`) en minimisant la somme des carrés des distances *géométriques* des points à l'ellipse. C'est une étape de finition qui améliore la précision de l'ajustement.

---

## 4. Cas d'usage et scripts d'exemples

Le dossier `examples/` fournit des scripts complets qui illustrent comment utiliser la bibliothèque pour résoudre des problèmes concrets.

### `case_shadow_triptych.py`

Ce script traite une image contenant plusieurs "panneaux" verticaux. Pour chaque panneau, il détecte deux ombres, mesure leur séparation, et ajuste une ellipse globale.

**Logique du pipeline :**
1.  **Découpage** : L'image est divisée en `n` panneaux verticaux (`split_equal_panels`).
2.  **Normalisation** : L'éclairage de chaque panneau est uniformisé pour corriger les gradients (`normalize_flatfield`).
3.  **Détection d'ombres** :
    - La méthode principale (`find_two_shadows_by_projection`) projette l'intensité d'une bande horizontale pour trouver les deux zones les plus sombres. C'est une méthode rapide et robuste.
    - En cas d'échec, une méthode de secours (`select_two_shadows`) est utilisée. Elle segmente l'image en composants connexes et sélectionne la meilleure paire d'ombres en se basant sur des critères de forme, de taille et de position.
4.  **Ajustement** : Une région d'intérêt (ROI) est définie autour des deux ombres. Les contours sont extraits de cette ROI, et une ellipse est ajustée avec `fit_ellipse_ransac`.
5.  **Sorties** : Un fichier CSV est généré avec les paramètres de l'ellipse pour chaque panneau, et des images de superposition sont sauvegardées pour le contrôle visuel.

### `case_two_shadow_average.py`

Ce script adopte une approche plus fidèle à la physique du problème : si les deux ombres proviennent du même objet, il est préférable de moyenner leurs formes avant l'ajustement final.

**Logique du pipeline :**
1.  **Détection des blobs** : Pour chaque panneau, les deux blobs les plus sombres sont détectés n'importe où dans l'image (`detect_two_blobs_anywhere`), sans contrainte de position.
2.  **Ajustement par blob** : Une ellipse est ajustée **individuellement** sur chaque blob.
3.  **Moyennage en coordonnées polaires** :
    - Pour chaque panneau, les contours des deux ellipses sont recentrés, convertis en coordonnées polaires (angle → rayon), et leurs rayons sont moyennés. Cela crée une forme moyenne `A_k` pour le panneau `k`.
    - Une ellipse est ensuite ajustée sur cette forme `A_k`.
4.  **Moyennage global** : Les formes moyennes `A_k` des 3 panneaux sont à leur tour moyennées en coordonnées polaires pour obtenir une forme moyenne globale `Ā`.
5.  **Ajustement final** : L'ellipse finale, représentant le diamètre de l'objet, est ajustée sur `Ā`.

Cette méthode est plus complexe mais potentiellement plus précise car elle exploite la redondance des informations (6 blobs au total) pour construire un modèle moyen robuste.

---

## 5. Guide d'utilisation

Voici un exemple de workflow de base pour utiliser la bibliothèque :

```python
import numpy as np
from elli_fit.io import load_binary_matrix, save_xy_csv
from elli_fit.core import edge_points, fit_ellipse_ransac, ellipse_points

# 1. Charger une image binarisée depuis un CSV
try:
    mask = load_binary_matrix("data/my_mask.csv")
except FileNotFoundError as e:
    print(e)
    exit()

# 2. Extraire les points du contour
#    Pour des données réelles, préférez edge_points_ordered si un seul contour est attendu.
x, y = edge_points(mask, threshold=0.5)

if len(x) < 20:
    print("Pas assez de points de contour pour un ajustement fiable.")
    exit()

# 3. Ajuster une ellipse de manière robuste
#    Les paramètres sont : (cx, cy, a, b, theta)
try:
    params = fit_ellipse_ransac(x, y, min_inliers=max(20, len(x) // 3))
except RuntimeError as e:
    print(f"L'ajustement RANSAC a échoué : {e}")
    exit()

# Optionnel : raffiner l'ajustement
# params = refine_ellipse_lm(x, y, *params)

# 4. Générer des points sur l'ellipse ajustée pour la visualisation
xp, yp = ellipse_points(*params, n=200)

# 5. Sauvegarder les points de l'ellipse ajustée
save_xy_csv("output/ellipse_fit.csv", xp, yp)

print("Ajustement terminé. Paramètres de l'ellipse :")
print(f"  Centre (cx, cy) = ({params[0]:.2f}, {params[1]:.2f})")
print(f"  Axes (a, b)   = ({params[2]:.2f}, {params[3]:.2f})")
print(f"  Angle (deg)     = {np.rad2deg(params[4]):.2f}°")
```

---

**Auteur** : Adrian Utge Le Gall, 2025



python3 /case_two_shadow_average.py  \
  --panels 3 --refine --save-dir out_case \
  --min-area 60 --alpha-dark 2.2 --thr-widen 0.22 \
  # ces 3 lignes exploitent les nouveaux filtres
  --border-margin 8 --max-bbox-frac 0.70 --min-roundness 0.45
