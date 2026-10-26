exec(open('c1.py').read()); exec(open('c8.py').read()); exec(open('c9.py').read())
import json
out={'hist':{k:v[1] for k,v in results.items()},'acc':{k:v[2] for k,v in results.items()},'att':{}}
for k in ['learned','none','rope']:
    m=results[k][0]
    with torch.no_grad(): m(torch.tensor([[3,1,4,1,5,9,2,6]]))
    out['att'][k]=[ [b.attn.last_attention[0].mean(0).tolist() for b in m.blocks] ]
import numpy as np
r=np.random.default_rng(0);x=r.integers(0,10,(200000,8))
mx=np.array([np.bincount(row,minlength=10).max() for row in x[:50000]])
out['bound']=float(mx.mean()/8)
json.dump(out,open('ch5_results.json','w'))
print(out['acc'],out['bound'])
