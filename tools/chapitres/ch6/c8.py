# CELLULE 8 — Mémoire : paramètres entraînés et états d'Adam, mesurés plutôt que supposés.
def optimizer_state_numel(model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = F.cross_entropy(model(X_tr[:8]), z_tr[:8]); loss.backward(); opt.step()
    return sum(v.numel() for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0)

for m in ['full', 'lora-4']:
    model = adapt(m, 0, steps=1)[0]
    print(f'{m:7s} entraînés : {trainable(model):7d} ; états Adam : {optimizer_state_numel(model):7d}')

def training_bytes(n_params, n_trainable, weight_bytes=2):
    """Poids (précision de stockage) + gradients, états Adam (m, v) et copie fp32 des paramètres entraînés."""
    return n_params * weight_bytes + n_trainable * (2 + 4 + 4 + 4)
N = 7e9
for label, wb, frac in [('ajustement complet, 16 bits', 2, 1.), ('LoRA, base 16 bits', 2, .005), ('QLoRA, base 4 bits', .5, .005)]:
    print(f'{label:30s} ≈ {training_bytes(N, frac * N, wb) / 2**30:6.1f} Gio (hors activations)')
