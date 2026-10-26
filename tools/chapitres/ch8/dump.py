import json
g={}
exec(''.join(open(f'c{i}.py').read()+'\n' for i in range(1,7)),g)
torch=g['torch']
xs=torch.linspace(-3,3,61);G=torch.stack(torch.meshgrid(xs,xs,indexing='xy'),-1).reshape(-1,2)
with torch.no_grad(): Eg=g['ebm'](G)
Eg=(Eg-Eg.min()).tolist()
sub=lambda s:[[round(a,3),round(b,3)] for a,b in s[:400].tolist()]
out=dict(energy=[round(v,3) for v in Eg],samples={k:sub(v) for k,v in g['table'].items()},
 table={k:dict(**{kk:vv for kk,vv in g['coverage'](v).items() if kk!='counts'},balance=g['balance'](v),spread=g['spread'](v)) for k,v in g['table'].items()},
 snaps={str(k):sub(v) for k,v in g['ddpm_snapshots'].items()},hist=[[s,round(a,3),round(b,3)] for s,a,b in g['ebm_history']])
json.dump(out,open('results_min.json','w'),ensure_ascii=False,separators=(',',':'))
print('ok',len(json.dumps(out)))
