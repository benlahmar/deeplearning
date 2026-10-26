# CELLULE 6 — Le principe de FlashAttention : softmax en ligne, par blocs de clés.
def attention_by_blocks(Q, K, V, block_size=4):
    n, dk = Q.shape
    m = np.full((n, 1), -np.inf)        # maximum courant des scores
    l = np.zeros((n, 1))                # somme courante des exponentielles
    O = np.zeros((n, V.shape[1]))       # accumulateur non normalisé
    for start in range(0, K.shape[0], block_size):
        Kb, Vb = K[start:start+block_size], V[start:start+block_size]
        S = Q @ Kb.T / np.sqrt(dk)
        m_new = np.maximum(m, S.max(1, keepdims=True))
        scale = np.exp(m - m_new)       # réajuste ce qui a déjà été accumulé
        E = np.exp(S - m_new)
        l = l * scale + E.sum(1, keepdims=True)
        O = O * scale + E @ Vb
        m = m_new
    return O / l

r = np.random.default_rng(3)
Q, K, V = r.normal(size=(5, 16)), r.normal(size=(23, 16)), r.normal(size=(23, 8))
full, _ = attention(Q, K, V)
for bs in (1, 4, 7, 23):
    assert np.allclose(attention_by_blocks(Q, K, V, bs), full)
print('Attention par blocs = attention complète, sans matrice n × m stockée.')
