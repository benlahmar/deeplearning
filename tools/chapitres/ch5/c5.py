# CELLULE 5 — Attention multi-têtes et bloc Transformer pré-normalisé (forward NumPy).
def layer_norm(X, g, b, eps=1e-5):
    mu = X.mean(-1, keepdims=True); var = X.var(-1, keepdims=True)
    return g * (X - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))

def init_block(d, seed=0):
    r = np.random.default_rng(seed); s = d**-.5
    return {'Wq': r.normal(0, s, (d, d)), 'Wk': r.normal(0, s, (d, d)),
            'Wv': r.normal(0, s, (d, d)), 'Wo': r.normal(0, s, (d, d)),
            'W1': r.normal(0, s, (d, 4*d)), 'c1': np.zeros(4*d),
            'W2': r.normal(0, (4*d)**-.5, (4*d, d)), 'c2': np.zeros(d),
            'g1': np.ones(d), 'b1': np.zeros(d), 'g2': np.ones(d), 'b2': np.zeros(d)}

def split_heads(T, h):          # (B, n, d) -> (B, h, n, d/h)
    B, n, d = T.shape
    return T.reshape(B, n, h, d // h).transpose(0, 2, 1, 3)

def merge_heads(T):             # (B, h, n, d_h) -> (B, n, h·d_h)
    B, h, n, dh = T.shape
    return T.transpose(0, 2, 1, 3).reshape(B, n, h * dh)

def mha(X, P, h, mask=None):
    Q, K, V = (split_heads(X @ P[w], h) for w in ('Wq', 'Wk', 'Wv'))
    O, A = attention(Q, K, V, mask)          # le masque (n, n) est diffusé sur B et h
    return merge_heads(O) @ P['Wo'], A

def block(X, P, h, mask=None):
    X = X + mha(layer_norm(X, P['g1'], P['b1']), P, h, mask)[0]
    H = gelu(layer_norm(X, P['g2'], P['b2']) @ P['W1'] + P['c1'])
    return X + H @ P['W2'] + P['c2']

d, h = 64, 4
P = init_block(d)
n_params = sum(v.size for v in P.values())
print('Paramètres du bloc :', n_params, '; formule 12d² + 9d =', 12*d*d + 9*d)
assert n_params == 12*d*d + 9*d
Xb = np.random.default_rng(2).normal(size=(2, 10, d))
Y = block(Xb, P, h, causal_mask(10))
print('Forme de sortie :', Y.shape)
