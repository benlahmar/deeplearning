# CELLULE 2 — Données : chiffres 8 × 8, sans étiquettes pour l'apprentissage auto-supervisé.
digits = load_digits()
X_all = torch.tensor(digits.images / 16., dtype=torch.float32)          # (1797, 8, 8) dans [0, 1]
y_all = torch.tensor(digits.target)
perm = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(0))
X_ssl, y_ssl = X_all[perm[:1200]], y_all[perm[:1200]]   # étiquettes ignorées pendant le préentraînement
X_test, y_test = X_all[perm[1200:]], y_all[perm[1200:]]
def draw_few(seed, per_class=5):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.nonzero(y_ssl == c)[:, 0][torch.randperm(int((y_ssl == c).sum()), generator=g)[:per_class]] for c in range(10)])
FEW = [draw_few(s) for s in range(5)]                    # cinq tirages de 5 étiquettes par classe
# Test « décalé » : chaque chiffre translaté d’un pixel en diagonale — l’invariance que les vues ont enseignée.
g_shift = torch.Generator().manual_seed(7)
shifts = 2 * torch.randint(0, 2, (len(X_test), 2), generator=g_shift) - 1
X_test_shift = torch.stack([torch.roll(x, (int(s[0]), int(s[1])), (0, 1)) for x, s in zip(X_test, shifts)])
print('Auto-supervisé :', len(X_ssl), '; sondes à 50 étiquettes (5 tirages) ; test :', len(X_test), '(+ version décalée)')

def augment(x, gen):
    """Deux vues d'un même chiffre : translation, effacement d'un bloc, intensité, bruit."""
    B = x.shape[0]
    dx, dy = torch.randint(-1, 2, (2, B), generator=gen)
    x = torch.stack([torch.roll(x[i], (int(dy[i]), int(dx[i])), (0, 1)) for i in range(B)])
    i0, j0 = torch.randint(0, 6, (2, B), generator=gen)
    erase = torch.rand(B, generator=gen) < .5
    for b in torch.nonzero(erase)[:, 0]:
        x[b, i0[b]:i0[b] + 3, j0[b]:j0[b] + 3] = 0.
    x = x * (.7 + .6 * torch.rand(B, 1, 1, generator=gen))
    return (x + .05 * torch.randn(x.shape, generator=gen)).clamp(0, 1)
