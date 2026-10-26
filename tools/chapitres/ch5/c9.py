# CELLULE 9 — Tâche : inverser une séquence. Sans information de position, est-ce possible ?
VOCAB, N = 10, 8

def batch(size, generator):
    x = torch.randint(0, VOCAB, (size, N), generator=generator)
    return x, x.flip(1)                      # y_i = x_{N-1-i}

def train(position, steps=1500, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(100 + seed)
    model = Encoder(VOCAB, N, position=position)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1., (t + 1) / 100))   # warmup
    history = []
    for step in range(steps):
        x, y = batch(128, g)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            history.append((step, loss.item()))
    model.eval()
    with torch.no_grad():
        x, y = batch(2000, torch.Generator().manual_seed(999))
        acc = (model(x).argmax(-1) == y).float().mean().item()
    return model, history, acc

results = {}
for position in ['none', 'sinusoidal', 'learned', 'rope']:
    t0 = time.time()
    results[position] = train(position)
    print(f'{position:10s} accuracy par jeton : {results[position][2]:.3f}   ({time.time()-t0:.0f} s)')
print('Hasard :', 1 / VOCAB)
