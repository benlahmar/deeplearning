import os
PAGE = 'ch6'
OUT = 'Chapitre_6_Transfert.html'
NCELLS = 11
TODOS = [
 (4, r'    def forward\(self, x\):\n', r'\n    def merged_weight', 'sortie de la couche gelée + (α/r)·(x Aᵀ) Bᵀ', '        '),
 (9, r'"""Quantification symétrique absmax, par tenseur \(group=None\) ou par groupes de colonnes\."""\n', r'\n\ndef quantized', 'échelle absmax par groupe, arrondi, bornage à ±(2^(b−1) − 1), retour à la forme de W'),
 (11, r'def distill\(alpha, T_kd=2.0, steps=1500, seed=0\):\n', r'\n\nprint\(', 'boucle d’entraînement de l’élève : (1 − α)·CE + α·T²·KL(maître ‖ élève) à température T'),
]
D = os.path.dirname(os.path.abspath(__file__)) + '/'
MD = {1: "# Chapitre 6 — Préentraînement, transfert et efficacité\n## Atelier {kind} · M2 IA\nPyTorch CPU, NumPy, Matplotlib. Prédire → mesurer → expliquer.\n",
 2: "## Atelier 1 · Un langage synthétique à quatre thèmes\nLe thème n’est jamais donné au modèle de langage.\n",
 3: "### Préentraîner un petit GPT\nPrédisez la perte de validation : plus proche de 1,860 ou de 1,520 ?\n",
 4: "## Atelier 2 · LoRA et tête de classification\nVérifiez qu’avec B = 0 le modèle adapté calcule exactement la même chose que le modèle préentraîné.\n",
 5: "### Cinq stratégies, 64 exemples, trois graines\nClassez les stratégies avant d’exécuter.\n",
 6: "### Oubli\nQue devient le modèle de langage après chaque adaptation ?\n",
 7: "## Atelier 3 · Efficacité en étiquettes (≈ 3 min)\n",
 8: "### Mémoire : ce qu’Adam stocke vraiment\n",
 9: "### Quantification après entraînement\n",
 10: "### Une valeur aberrante\n",
 11: "## Atelier 4 · Distillation (≈ 1 min)\nRefaites l’expérience sur plusieurs graines avant de conclure.\n"}
DATA = {'@@RESULTS@@': open(D + 'results_min.json', encoding='utf-8').read().strip()}
REPLACE = [('<b>Fig. 6.6 —</b>', '<b>Fig. 6.5 —</b>'), ('<b>Fig. 6.7 —</b>', '<b>Fig. 6.6 —</b>'), ('<b>Fig. 6.8 —</b>', '<b>Fig. 6.7 —</b>'), ('<b>Fig. 6.9 —</b>', '<b>Fig. 6.8 —</b>')]
