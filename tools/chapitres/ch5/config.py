import os, json
PAGE = 'ch5'
OUT = 'Chapitre_5_Transformers.html'
NCELLS = 10
RAISE_SHORT = True   # message du NotImplementedError : partie avant « : »
TODOS = [
 (2, r'def masked_softmax\(S, mask=None\):\n', r'\n\ndef attention', 'softmax ligne par ligne, poids nuls hors masque, lignes vides → 0 sans NaN'),
 (2, r'"""Q : \(\.\.\., n, d_k\) ; K : \(\.\.\., m, d_k\) ; V : \(\.\.\., m, d_v\) ; mask : \(\.\.\., n, m\)\."""\n', r'\n\ndef causal_mask', 'scores QKᵀ/√d_k, softmax masqué, retourner (A @ V, A)'),
 (6, r'def attention_by_blocks\(Q, K, V, block_size=4\):\n', r'\n\nr = np.random', 'parcourir K, V par blocs en tenant à jour m, l et O (TD, exercice 7)'),
 (7, r'def mha_incremental\(X, P, h\):\n', r'\n\nXs = ', 'un jeton à la fois : requête courante, cache des clés et valeurs, attention sans masque'),
 (8, r'"""Rotation des paires de coordonnées : x de forme \(B, h, n, d_h\)\."""\n', r'\n\nclass MHA', 'tourner chaque paire (x[..., 2i], x[..., 2i+1]) d’un angle position × θ_i'),
]
MD = {1: "# Chapitre 5 — Attention et Transformers\n## Atelier {kind} · M2 IA\nNumPy et Matplotlib pour les ateliers 1 et 2 ; PyTorch CPU pour l’atelier 3.\nPrédire → vérifier → expliquer.\n",
 2: "## Atelier 1 · Attention masquée\nUne ligne entièrement masquée ne doit pas produire de NaN.\n",
 3: "### Trois propriétés à vérifier\nPrédisez le résultat de chaque test avant de l’exécuter.\n",
 4: "### L’effet de 1/√d_k\nNotez votre prédiction pour l’entropie à d_k = 1024 sans normalisation.\n",
 5: "## Atelier 2 · Bloc Transformer pré-normalisé\nVérifiez le nombre de paramètres avant de calculer quoi que ce soit.\n",
 6: "### Softmax en ligne (principe de FlashAttention)\nLa preuve est l’exercice 7 du TD.\n",
 7: "### Décodage avec cache KV\nPourquoi n’y a-t-il pas de masque dans la version incrémentale ?\n",
 8: "## Atelier 3 · Un encodeur PyTorch et quatre options de position\n",
 9: "### Inverser une séquence\nPrédisez les quatre accuracies (TD, exercice 3) avant l’entraînement.\n",
 10: "### Courbes et cartes d’attention\nCherchez l’anti-diagonale. Comparez les lignes 1 et 3 du modèle sans position.\n"}
D = os.path.dirname(os.path.abspath(__file__)) + '/'
# Figure « effet de 1/√d_k » : [d_k, Var(q·k), H brut, H normalisé, ‖J‖ brut, ‖J‖ normalisé]
SCALE = [[2,1.87,2.8036,3.0795,0.2557,0.2276],[4,3.9,2.2946,3.0433,0.2919,0.2338],[8,8.16,1.6487,3.0057,0.3079,0.2394],[16,15.67,1.1352,3.0203,0.2932,0.2398],[32,31.98,0.7582,3.0127,0.2438,0.2408],[64,63.64,0.4939,3.0047,0.1979,0.2422],[128,131.06,0.3301,3.0013,0.1502,0.2419],[256,260.32,0.2339,3.0092,0.1148,0.2418],[512,520.71,0.1597,3.0107,0.0763,0.2415],[1024,1036.22,0.1162,3.0105,0.0662,0.2415]]
DATA = {'@@RESULTS@@': open(D + 'results_min.json', encoding='utf-8').read().strip(), '@@SCALE@@': json.dumps(SCALE)}
REPLACE = []
