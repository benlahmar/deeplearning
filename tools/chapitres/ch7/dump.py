import json
g={}
exec(''.join(open(f'c{i}.py').read()+'\n' for i in range(1,8)),g)
out=dict(results={k:{kk:(vv[:32] if kk=='spectrum' else vv) for kk,vv in v.items()} for k,v in g['results'].items()},histories=g['histories'],seq=g['seq_results'])
json.dump(out,open('ch7_results.json','w'))
print('ok')
