import os
PAGE = 'ch8'
OUT = 'Chapitre_8_Energie.html'
NCELLS = 6
TODOS = [
 (3, r'"""x ← x − \(ε/2\) ∇E\(x\) \+ √ε ξ : une marche qui descend l\'énergie en restant aléatoire\."""\n', r'\n\ndef train_ebm', 'steps pas : gradient de E (autograd), demi-pas de descente, bruit √ε·ξ si noise ; retourner x détaché'),
 (4, r'def train_dsm\(steps=4000, seed=0\):\n', r'\n\n@torch\.no_grad\(\)', 'bruiter x à un σ tiré dans SIGMAS, cible −bruit/σ, perte σ²‖s_θ − cible‖² ; retourner (modèle, historique)'),
 (5, r'def ddpm_sample\(net, n=2000, seed=1, keep=\(199, 150, 100, 50, 0\)\):\n', r'\n\nt0 = time\.time\(\)', 'partir de N(0, I), puis pour t = T−1 … 0 : retirer le bruit prédit, diviser par √α_t, ajouter √β_t·ξ si t > 0 ; retourner (x, instantanés)'),
]
D = os.path.dirname(os.path.abspath(__file__)) + '/'
MD = {1: "# Chapitre 8 — Modèles à énergie et ouverture recherche\n## Atelier {kind} · M2 IA\nPyTorch CPU, Matplotlib. Une densité 2D connue pour juger trois générateurs.\n",
 2: "## Atelier 1 · Huit gaussiennes sur un cercle, et deux mesures\n",
 3: "### Modèle à énergie : maximum de vraisemblance et négatifs par Langevin (≈ 1 min)\n",
 4: "## Atelier 2 · Score matching débruité et Langevin recuit\n",
 5: "### Diffusion (DDPM) : prédire le bruit\n",
 6: "## Atelier 3 · Contre-expériences\nPrédisez qualité, couverture et dispersion avant d’exécuter.\n"}
DATA = {'@@RESULTS@@': open(D + 'results_min.json', encoding='utf-8').read().strip()}
REPLACE = []
