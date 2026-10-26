# CELLULE 7 — Décodage avec cache KV : même résultat, beaucoup moins de calcul.
def mha_full_causal(X, P, h):
    return mha(X[None], P, h, causal_mask(len(X)))[0][0]

def mha_incremental(X, P, h):
    cache_k, cache_v, outputs = [], [], []
    for t in range(len(X)):
        x = X[t:t+1][None]                              # un seul nouveau jeton
        q = split_heads(x @ P['Wq'], h)
        cache_k.append(split_heads(x @ P['Wk'], h)); cache_v.append(split_heads(x @ P['Wv'], h))
        K, V = np.concatenate(cache_k, axis=2), np.concatenate(cache_v, axis=2)
        o, _ = attention(q, K, V)                       # pas de masque : le cache ne contient que le passé
        outputs.append(merge_heads(o) @ P['Wo'])
    return np.concatenate(outputs, axis=1)[0]

Xs = np.random.default_rng(4).normal(size=(12, d))
assert np.allclose(mha_full_causal(Xs, P, h), mha_incremental(Xs, P, h))
print('Décodage incrémental avec cache = passage causal complet.')

def kv_cache_bytes(layers, n_kv_heads, d_head, seq, batch=1, bytes_per=2):
    return 2 * layers * n_kv_heads * d_head * seq * batch * bytes_per
for name, kv in [('MHA (32 têtes K/V)', 32), ('GQA (8 groupes)', 8), ('MQA (1 tête K/V)', 1)]:
    print(f'{name:20s} : {kv_cache_bytes(32, kv, 128, 8192) / 2**30:.2f} Gio pour 8 192 jetons')
