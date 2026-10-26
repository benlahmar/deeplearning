# CELLULE 2 — Attention produit scalaire normalisée, avec masque booléen (True = autorisé).
def masked_softmax(S, mask=None):
    if mask is None:
        mask = np.ones(S.shape, dtype=bool)
    S = np.where(mask, S, -np.inf)
    m = S.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.)          # ligne entièrement masquée : pas de NaN
    E = np.where(mask, np.exp(S - m), 0.)
    Z = E.sum(axis=-1, keepdims=True)
    return np.divide(E, Z, out=np.zeros_like(E), where=Z > 0)

def attention(Q, K, V, mask=None):
    """Q : (..., n, d_k) ; K : (..., m, d_k) ; V : (..., m, d_v) ; mask : (..., n, m)."""
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    A = masked_softmax(S, mask)
    return A @ V, A

def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))

rng = np.random.default_rng(0)
n, d = 6, 8
X = rng.normal(size=(n, d))
Wq, Wk, Wv = (rng.normal(0, d**-.5, (d, d)) for _ in range(3))
out, A = attention(X @ Wq, X @ Wk, X @ Wv, causal_mask(n))
print('Poids causaux (ligne i = requête i) :\n', A)
assert np.allclose(A.sum(-1), 1) and np.allclose(np.triu(A, 1), 0)
