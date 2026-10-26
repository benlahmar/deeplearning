# CELLULE 2 — Une densité 2D connue : huit gaussiennes sur un cercle.
MODES = torch.tensor([[2 * math.cos(2 * math.pi * k / 8), 2 * math.sin(2 * math.pi * k / 8)] for k in range(8)])
SIGMA = .15
def sample_data(n, gen):
    k = torch.randint(0, 8, (n,), generator=gen)
    return MODES[k] + SIGMA * torch.randn(n, 2, generator=gen)

def coverage(samples):
    """Qualité : part des échantillons à moins de 3σ d'un mode. Couverture : modes recevant au moins 2 %."""
    d = torch.cdist(samples, MODES)
    near, k = d.min(1)
    good = near < 3 * SIGMA
    counts = torch.bincount(k[good], minlength=8).float() / len(samples)
    return dict(quality=good.float().mean().item(), modes=int((counts > .02).sum()), counts=counts.tolist())

gen = torch.Generator().manual_seed(0)
X = sample_data(4000, gen)
print('Données :', tuple(X.shape), '; contrôle sur les données elles-mêmes :', coverage(X)['quality'], coverage(X)['modes'])
