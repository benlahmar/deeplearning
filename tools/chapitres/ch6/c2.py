# CELLULE 2 — Un « langage » synthétique : quatre thèmes, chacun avec ses transitions.
V, T, K = 24, 32, 4                       # vocabulaire, longueur, nombre de thèmes
g = torch.Generator().manual_seed(0)
base = 2.0 * torch.randn(V, V, generator=g)           # structure commune à tous les thèmes
TRANS = (base + 1.5 * torch.randn(K, V, V, generator=g)).softmax(-1)   # (K, V, V) : un écart propre à chaque thème

def sample(n, seed):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randint(0, K, (n,), generator=gen)
    x = torch.empty(n, T, dtype=torch.long)
    x[:, 0] = torch.randint(0, V, (n,), generator=gen)
    for t in range(1, T):
        x[:, t] = torch.multinomial(TRANS[z, x[:, t - 1]], 1, generator=gen).squeeze(1)
    return x, z

X_pre, _ = sample(20000, seed=1)          # préentraînement : aucune étiquette utilisée
X_lm_val, _ = sample(2000, seed=2)        # mesure de la perte de langage (oubli)
X_tr, z_tr = sample(64, seed=3)           # tâche cible : 64 exemples étiquetés
X_te, z_te = sample(4000, seed=4)
print('Préentraînement', tuple(X_pre.shape), '; cible', tuple(X_tr.shape), '; test', tuple(X_te.shape))
