# CELLULE 9 — Quantification après entraînement : arrondir les poids et mesurer.
def fake_quant(W, bits, group=None):
    """Quantification symétrique absmax, par tenseur (group=None) ou par groupes de colonnes."""
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(-1, group) if group else W.reshape(1, -1)
    scale = Wg.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(Wg / scale).clamp(-qmax, qmax) * scale).reshape(W.shape)

def quantized(model, bits, group=None):
    q = copy.deepcopy(model)
    with torch.no_grad():
        for blk in q.blocks:
            for lin in (blk.qkv, blk.proj, blk.fc1, blk.fc2):
                lin.weight.copy_(fake_quant(lin.weight, bits, group))
    return q

print(f'{"référence 32 bits":24s} perte LM {lm_loss(pretrained, X_lm_val):.4f}')
for bits, group in [(8, None), (4, None), (4, 32), (3, None), (3, 32)]:
    name = f'int{bits} ' + ('par tenseur' if group is None else f'groupes de {group}')
    print(f'{name:24s} perte LM {lm_loss(quantized(pretrained, bits, group), X_lm_val):.4f}')
