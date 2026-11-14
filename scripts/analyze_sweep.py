from pathlib import Path
import json,hashlib,statistics
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Aggregate D sweep receipts; query-bootstrap diagnostic intervals')
parser.add_argument('--run-dir',required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--allow-partial',action='store_true')
a=parser.parse_args();run=Path(a.run_dir);out=Path(a.out)
state=json.loads((run/'state.json').read_text())
if state['status']!='complete' and not a.allow_partial:parser.error('Run is incomplete; inspect it or explicitly use --allow-partial')
if out.exists():parser.error('Output exists; use a new filename')
files=sorted((run/'results').glob('*.json'));runs={p.stem:json.loads(p.read_text()) for p in files}
groups={}
for name,r in runs.items():groups.setdefault(name.rsplit('-s',1)[0],[]).append(r)
def macro_ci(rs,method,key):
 per=[]
 for r in rs:
  by={}
  for v in r['per_query'][method]:
   if v.get(key) is not None:by.setdefault(v['query'],[]).append(v[key])
  per.append({q:float(np.mean(vals)) for q,vals in by.items()})
 common=sorted(set.intersection(*(set(x) for x in per)))
 if not common:return None
 vals=np.array([np.mean([p[q] for p in per]) for q in common]);rng=np.random.default_rng(20261114)
 boot=vals[rng.integers(0,len(vals),size=(2000,len(vals)))].mean(axis=1)
 return dict(mean=float(vals.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),queries=len(vals))
aggregate={}
for name,rs in groups.items():
 assert len({r['workload_sha256'] for r in rs})==1,(name,'data changed with index seed')
 assert all(all(v['eligibility_violations']==v['group_violations']==0 for v in r['summary'].values()) for r in rs)
 methods={}
 exact=min((k for k in rs[0]['summary'] if k.startswith('exact_')),key=lambda k:statistics.median(r['summary'][k]['p50_ms'] for r in rs))
 for method in rs[0]['summary']:
  vals=[r['summary'][method] for r in rs]
  methods[method]={'group_recall':macro_ci(rs,method,'group_recall'),'p50_ms_median_seeds':statistics.median(v['p50_ms'] for v in vals),
    'p95_ms_median_seeds':statistics.median(v['p95_ms'] for v in vals),'speedup_median_seeds':statistics.median(r['summary'][exact]['p50_ms']/r['summary'][method]['p50_ms'] for r in rs),
    'recall_range_seeds':[min(v['group_recall'] for v in vals),max(v['group_recall'] for v in vals)],'fill_mean':statistics.mean(v['fill'] for v in vals)}
  for key in ['evidence_recall','positive_recall','contradict_recall']:
   ci=macro_ci(rs,method,key)
   if ci:methods[method][key]=ci
 aggregate[name]={'seeds':[r['seed'] for r in rs],'n':rs[0]['n'],'d':rs[0]['d'],'queries':rs[0]['queries'],'exact_baseline':exact,
   'workload_sha256':rs[0]['workload_sha256'],'methods':methods,
   'build_seconds_median':statistics.median(r['build_seconds'] for r in rs),
   'peak_rss_gib_max':max(r['process_peak_rss_kib'] for r in rs)/1024**2,
   'serialized_index_gib_median':statistics.median(r['serialized_index_bytes'] for r in rs)/1024**3}
summary={'manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),'run_status':state['status'],'cases':len(runs),'groups':aggregate,
 'interval_method':'2000 bootstrap samples of queries; each query averages the three observed index seeds. Conditional on these fixed representations/queries/seeds, not a general performance guarantee.',
 'receipts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as handle:json.dump(summary,handle,indent=2,allow_nan=False)
for name,g in aggregate.items():
 print(name,'n',g['n'],'seeds',g['seeds'],'peak',round(g['peak_rss_gib_max'],2),'build',round(g['build_seconds_median'],2))
 for method in ['exact_faiss','hnsw_adaptive_ef128','hnsw_adaptive_ef256','hnsw_filtered_ef256','hnsw_collapsed_ef256']:
  if method not in g['methods']:continue
  m=g['methods'][method];print(' ',method,'recall',m['group_recall'],'speed',round(m['speedup_median_seeds'],2),'p50',round(m['p50_ms_median_seeds'],3),'fill',round(m['fill_mean'],3))
