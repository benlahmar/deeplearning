import os
PAGE = 'ch7'
OUT = 'Chapitre_7_SSL_JEPA.html'
NCELLS = 7
TODOS = [
 (4, r'def info_nce\(z1, z2, tau=\.2\):\n', r'\n\ndef vicreg', 'normaliser, similarités / τ, exclure la diagonale, entropie croisée vers la vue jumelle'),
 (4, r'def vicreg\(z1, z2, lam=25\., mu=25\., nu=1\., use_variance=True\):\n', r'\n\ndef train_joint', 'invariance (MSE) + variance (hinge sur l’écart-type) + covariance (hors diagonale), sur les deux branches'),
 (5, r'def ema_update\(target, online, m\):\n', r'\n\ndef train_byol', 'ξ ← m·ξ + (1 − m)·θ pour chaque paramètre, sans gradient'),
]
D = os.path.dirname(os.path.abspath(__file__)) + '/'
MD = {1: "# Chapitre 7 — Apprentissage auto-supervisé et JEPA\n## Atelier {kind} · M2 IA\nPyTorch CPU, scikit-learn (jeu digits inclus), Matplotlib.\n",
 2: "## Atelier 1 · Données, vues et test décalé\n",
 3: "### Encodeur et trois mesures : sonde, k-NN, rang effectif\n",
 4: "## Atelier 2 · Invariance, InfoNCE, VICReg\nPrédisez lequel de ces objectifs s’effondre (TD, exercice 3).\n",
 5: "### Sans négatifs : BYOL et une mini I-JEPA\n",
 6: "## Atelier 3 · Évaluation commune (≈ 4 min)\nComparez aussi au test décalé d’un pixel.\n",
 7: "## Atelier 4 · Une JEPA pour des séquences\nLe langage à quatre thèmes du chapitre 6 : prédire la représentation du futur, pas ses jetons.\n"}
DATA = {'@@RESULTS@@': open(D + 'results_min.json', encoding='utf-8').read().strip(), '@@DIGITS@@': open(D + 'digits_sample.json').read().strip()}
REPLACE = []
