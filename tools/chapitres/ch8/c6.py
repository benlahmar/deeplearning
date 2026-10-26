# CELLULE 6 — Deux contre-expériences : inférer n'est pas échantillonner ; un seul niveau de bruit ne suffit pas.
def spread(samples):
    """Écart-type moyen autour du mode le plus proche (données : σ = 0,15)."""
    d = torch.cdist(samples, MODES); k = d.argmin(1)
    return float((samples - MODES[k]).std(0).mean())

def balance(samples):
    c = torch.tensor(coverage(samples)['counts']); c = c / c.sum()
    return float(-(c * torch.log(c + 1e-12)).sum() / math.log(8))      # 1 = modes équilibrés

gen = torch.Generator().manual_seed(1)
start = 6 * torch.rand(2000, 2, generator=gen) - 3
descent = langevin(ebm, start, 300, .01, noise=False)                   # inférence : argmin E
SIGMAS_SAVE = SIGMAS
SIGMAS = SIGMAS_SAVE[-1:]                                               # un seul niveau : σ = 0,01
single, _ = train_dsm()
single_samples = annealed_langevin(single, T=1000)
SIGMAS = SIGMAS_SAVE

table = {'données': X[:2000], 'EBM, Langevin': ebm_samples, 'EBM, descente sans bruit': descent,
         'score, un seul σ': single_samples, 'score, σ recuit': dsm_samples, 'diffusion (DDPM)': ddpm_samples}
print(f'{"":26s} qualité  modes  équilibre  dispersion')
for name, s in table.items():
    c = coverage(s)
    print(f'{name:26s} {c["quality"]:.3f}    {c["modes"]}      {balance(s):.3f}      {spread(s):.3f}')
