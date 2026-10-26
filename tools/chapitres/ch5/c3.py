# CELLULE 3 — Trois propriétés à vérifier plutôt qu'à supposer.
def self_attention(X, mask=None):
    return attention(X @ Wq, X @ Wk, X @ Wv, mask)[0]

# (a) Causalité : modifier le jeton 4 ne change pas les sorties 0 à 3.
X2 = X.copy(); X2[4] += 10 * rng.normal(size=d)
diff = np.abs(self_attention(X, causal_mask(n)) - self_attention(X2, causal_mask(n))).max(axis=1)
print('Écart par position après modification du jeton 4 :', diff)
assert np.allclose(diff[:4], 0) and diff[4] > 0

# (b) Sans masque ni position, l'auto-attention est équivariante par permutation.
P = np.eye(n)[rng.permutation(n)]
assert np.allclose(self_attention(P @ X), P @ self_attention(X))

# (c) Le masque causal rompt cette équivariance : l'ordre compte à nouveau.
print('Équivariance avec masque causal :',
      np.allclose(self_attention(P @ X, causal_mask(n)), P @ self_attention(X, causal_mask(n))))

# (d) Masque de padding : les deux dernières positions sont du remplissage.
valid = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
pad_mask = valid[None, :] & valid[:, None]
_, A_pad = attention(X @ Wq, X @ Wk, X @ Wv, pad_mask)
assert np.allclose(A_pad[:, 4:], 0) and np.allclose(A_pad[4:], 0)
print('Lignes de padding (entièrement masquées) :', A_pad[4:].sum(1), '— zéros, pas de NaN')
