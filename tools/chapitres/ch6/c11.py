# CELLULE 11 — Distillation : un élève deux fois plus étroit apprend des probabilités du maître.
def distill(alpha, T_kd=2.0, steps=1500, seed=0):
    torch.manual_seed(seed)
    student = TinyGPT(d=32, h=2, layers=1)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=.01)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        xb = X_pre[torch.randint(0, len(X_pre), (64,), generator=gen)]
        s = student(xb[:, :-1]).reshape(-1, V)
        hard = F.cross_entropy(s, xb[:, 1:].reshape(-1))
        with torch.no_grad():
            t = pretrained(xb[:, :-1]).reshape(-1, V)
        soft = F.kl_div(F.log_softmax(s / T_kd, -1), F.softmax(t / T_kd, -1), reduction='batchmean') * T_kd**2
        loss = (1 - alpha) * hard + alpha * soft
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return student

print('Paramètres maître / élève :', sum(p.numel() for p in pretrained.parameters()),
      sum(p.numel() for p in TinyGPT(d=32, h=2, layers=1).parameters()))
for alpha in [0., .5, .9]:
    print(f'α = {alpha:.1f} : perte LM de l’élève {lm_loss(distill(alpha), X_lm_val):.3f}')
