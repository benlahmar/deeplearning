# CELLULE 4 — Score matching débruité à plusieurs niveaux de bruit, puis Langevin recuit.
SIGMAS = torch.exp(torch.linspace(math.log(1.), math.log(.01), 10))     # σ₁ > … > σ_L

class Score(nn.Module):
    """s_θ(x, σ) ≈ ∇ₓ log p_σ(x) ; le niveau de bruit est une entrée du réseau."""
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, sigma):
        return self.net(torch.cat([x, torch.log(sigma)], 1)) / sigma     # paramétrisation usuelle : sortie / σ

def train_dsm(steps=4000, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    S = Score(); opt = torch.optim.Adam(S.parameters(), lr=1e-3)
    history = []
    for step in range(steps):
        x = sample_data(512, gen)
        sigma = SIGMAS[torch.randint(0, len(SIGMAS), (512,), generator=gen)][:, None]
        noise = torch.randn(x.shape, generator=gen)
        x_tilde = x + sigma * noise
        target = -noise / sigma                                         # = ∇ log p(x̃ | x)
        loss = (sigma ** 2 * (S(x_tilde, sigma) - target) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            history.append((step, loss.item()))
    return S, history

@torch.no_grad()
def annealed_langevin(S, n=2000, eps=2e-4, T=100, seed=1):
    gen = torch.Generator().manual_seed(seed)
    x = 6 * torch.rand(n, 2, generator=gen) - 3
    for sigma in SIGMAS:
        a = eps * (sigma / SIGMAS[-1]) ** 2                             # pas proportionnel à σ²
        s = torch.full((n, 1), float(sigma))
        for _ in range(T):
            x = x + .5 * a * S(x, s) + torch.sqrt(a) * torch.randn(x.shape, generator=gen)
    return x

t0 = time.time()
score_model, dsm_history = train_dsm()
dsm_samples = annealed_langevin(score_model)
print(f'Score matching ({time.time()-t0:.0f} s) :', {k: v for k, v in coverage(dsm_samples).items() if k != 'counts'})
