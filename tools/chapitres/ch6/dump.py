import json
src=''.join(open(f'c{i}.py').read()+'\n' for i in range(1,12))
g={}
exec(src,g)
out=dict(pre=g['pre_history'],methods={m:dict(acc=v['acc'],params=v['params']) for m,v in g['results'].items()},
 forgetting=g['forgetting'],curve=g['label_curve'])
json.dump(out,open('ch6_results.json','w'))
print('ok')
