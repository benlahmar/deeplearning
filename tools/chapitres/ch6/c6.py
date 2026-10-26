# CELLULE 6 — Oubli : que devient le modèle de langage après adaptation ?
def lm_loss_of(model):
    trunk = model.trunk
    with torch.no_grad():
        logits = trunk.lm_head(trunk.features(X_lm_val[:, :-1]))   # même tête LM, tronc adapté
        return F.cross_entropy(logits.reshape(-1, V), X_lm_val[:, 1:].reshape(-1)).item()

forgetting = {'préentraîné': lm_loss(pretrained, X_lm_val)}
for m in ['probe', 'full', 'lora-4']:
    forgetting[m] = lm_loss_of(results[m]['model'])
lora_layers = [mod for mod in results['lora-4']['model'].modules() if isinstance(mod, LoRALinear)]
scales = [mod.scale for mod in lora_layers]
for mod in lora_layers:
    mod.scale = 0.                                   # on retire B·A
forgetting['lora-4, adaptateurs retirés'] = lm_loss_of(results['lora-4']['model'])
for mod, s in zip(lora_layers, scales):
    mod.scale = s
for k, v in forgetting.items():
    print(f'{k:28s} perte LM : {v:.3f}')
