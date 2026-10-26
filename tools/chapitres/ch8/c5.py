# CELLULE 5 — Diffusion (DDPM) : prédire le bruit, c'est estimer le score.
T_STEPS = 200
betas = torch.linspace(1e-4, .04, T_STEPS)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)

class EpsNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T_STEPS)[:, None]], 1))

def train_ddpm(steps=4000, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    net = EpsNet(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(steps):
        x0 = sample_data(512, gen)
        t = torch.randint(0, T_STEPS, (512,), generator=gen)
        eps = torch.randn(x0.shape, generator=gen)
        xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
        loss = F.mse_loss(net(xt, t), eps)
        opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def ddpm_sample(net, n=2000, seed=1, keep=(199, 150, 100, 50, 0)):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen); snapshots = {}
    for t in reversed(range(T_STEPS)):
        tt = torch.full((n,), t)
        eps = net(x, tt)
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn(x.shape, generator=gen)
        if t in keep:
            snapshots[t] = x.clone()
    return x, snapshots

t0 = time.time()
ddpm = train_ddpm()
ddpm_samples, ddpm_snapshots = ddpm_sample(ddpm)
print(f'DDPM ({time.time()-t0:.0f} s) :', {k: v for k, v in coverage(ddpm_samples).items() if k != 'counts'})
