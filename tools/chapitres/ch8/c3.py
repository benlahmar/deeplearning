# CELLULE 3 — Un modèle à énergie appris par maximum de vraisemblance, négatifs par Langevin.
class Energy(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def langevin(E, x, steps, step_size, noise=True, gen=None):
    """x ← x − (ε/2) ∇E(x) + √ε ξ : une marche qui descend l'énergie en restant aléatoire."""
    x = x.clone().requires_grad_(True)
    for _ in range(steps):
        g, = torch.autograd.grad(E(x).sum(), x)
        x = x - .5 * step_size * g
        if noise:
            x = x + math.sqrt(step_size) * torch.randn(x.shape, generator=gen)
        x = x.detach().requires_grad_(True)
    return x.detach()

def train_ebm(steps=3000, seed=0, k_mcmc=30, step_size=.01, alpha=.1):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    E = Energy(); opt = torch.optim.Adam(E.parameters(), lr=1e-3)
    buffer = 6 * torch.rand(8192, 2, generator=gen) - 3          # tampon persistant de négatifs
    history = []
    for step in range(steps):
        pos = sample_data(256, gen)
        idx = torch.randint(0, len(buffer), (256,), generator=gen)
        init = buffer[idx]
        fresh = torch.rand(256, generator=gen) < .05                  # 5 % de redémarrages uniformes
        init[fresh] = 6 * torch.rand(int(fresh.sum()), 2, generator=gen) - 3
        neg = langevin(E, init, k_mcmc, step_size, gen=gen)
        buffer[idx] = neg
        e_pos, e_neg = E(pos), E(neg)
        # Gradient de la log-vraisemblance : abaisser l'énergie des données, relever celle des échantillons du modèle.
        loss = e_pos.mean() - e_neg.mean() + alpha * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            history.append((step, e_pos.mean().item(), e_neg.mean().item()))
    return E, history

t0 = time.time()
ebm, ebm_history = train_ebm()
gen = torch.Generator().manual_seed(1)
ebm_samples = langevin(ebm, 6 * torch.rand(2000, 2, generator=gen) - 3, 300, .01, gen=gen)
print(f'EBM ({time.time()-t0:.0f} s) :', {k: v for k, v in coverage(ebm_samples).items() if k != 'counts'})
