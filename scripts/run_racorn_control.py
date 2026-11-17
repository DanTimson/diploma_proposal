#!/usr/bin/env python3
"""D-005 filter-only control using one pinned upstream graph and driver."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diploma_starters.group_screen import ROOT,digest,fvecs,write_json
from diploma_starters.assets import fetch

REV='b1a587ad2f5d4048cf50c8e2710e5070864590c3'
SUBREV='09aaff9d20dadbdbe133d3aba41b3c768c49888a'

def write_fvecs(path,x):
    rows=np.empty((len(x),x.shape[1]+1),dtype='<i4');rows[:,0]=x.shape[1];rows[:,1:]=np.asarray(x,dtype='<f4').view('<i4');rows.tofile(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['smoke','pilot'])
    p.add_argument('--n',type=int,choices=[100000,1000000],default=100000)
    p.add_argument('--upstream',default='external/RACORN');p.add_argument('--data-dir',default='external/sift1m')
    p.add_argument('--out',required=True);p.add_argument('--timeout',type=int,default=7200);a=p.parse_args()
    up=Path(a.upstream).resolve();binary=up/'build/bench/bench_synthfilter';out=Path(a.out).resolve()
    for path,rev in [(up,REV),(up/'third_party/NumKong',SUBREV)]:
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=rev:raise ValueError('Wrong upstream revision')
        if subprocess.check_output(['git','-C',str(path),'diff','HEAD','--'],text=True).strip():raise ValueError('Modified upstream source')
    build=json.loads((up/'build/bench/build-receipt.json').read_text())
    if digest(binary)!=build['binary_sha256'] or any(digest(up/name)!=h for name,h in build['sources'].items()):raise ValueError('Stale binary or source; rebuild')
    out.mkdir(parents=True,exist_ok=False);raw=out/'raw';raw.mkdir()
    write_json(out/'config.json',vars(a))
    if a.mode=='smoke':
        n,d,nq,k=1000,16,8,10;rng=np.random.default_rng(20261116)
        x=rng.normal(size=(n,d)).astype('f4');q=(x[:nq]+.1*rng.normal(size=(nq,d))).astype('f4')
        write_fvecs(raw/'sift_base.fvecs',x);qids=list(range(nq));selectivity='10%,50%';efs='32,128';runs='1'
        scope='Synthetic loader smoke only; not SIFT data'
    else:
        n,d,nq,k=a.n,128,32,100
        lock=json.loads((ROOT/'topics/D_hnsw/d003/sources.json').read_text())['sift']
        for name in ['sift_base.fvecs','sift_query.fvecs']:fetch(lock[name],Path(a.data_dir)/name)
        (raw/'sift_base.fvecs').symlink_to((Path(a.data_dir)/'sift_base.fvecs').resolve())
        split=json.loads((ROOT/'receipts/frozen-workload/manifest.json').read_text())
        qids=[split['fresh_query_ids'][i] for i in split['tune_indices']]
        if len(qids)!=32 or set(qids)&set(split['old_query_ids']):raise ValueError('Unexpected development split')
        q=np.ascontiguousarray(fvecs(Path(a.data_dir)/'sift_query.fvecs')[qids])
        selectivity='1%,10%,50%';efs='100,200,400,800,1600';runs='3'
        scope='SIFT prefix base, D004 development queries only; filter-only diagnostic, NOT grouped retrieval or paper reproduction'
    write_fvecs(raw/'sift_query.fvecs',q)
    cmd=[str(binary),'--dataset','sift','--n',str(n),'--d',str(d),'--nq',str(nq),'--k',str(k),
         '--m','16','--ef-construction','100','--build-threads','1','--search-threads','1',
         '--filter','allowlist','--seed','12345','--alpha','0,-0.8','--selectivity',selectivity,
         '--modes','prefilter,prefilter-scan,hnsw,acorn-1,racorn-1','--ef-values',efs,'--recall-target','1.0',
         '--bridge-budget-ratios','2.0','--filter-first-threshold','0.3','--stride','on','--ce','on',
         '--runs',runs,'--warmup','1','--data-dir',str(raw),'--volume-dir',str(out/'volumes'),
         '--gt-dir',str(out/'gt'),'--json',str(out/'upstream.json')]
    write_json(out/'manifest.json',dict(scope=scope,query_ids=qids,source_sha256=digest(__file__),
        upstream=REV,submodule=SUBREV,build=build,command=cmd,
        raw_hashes={p.name:digest(p) for p in raw.iterdir()},
        decision='Development screen: compare fastest measured configurations reaching recall >=0.95 in each cell; no successful configuration means fail at tested budgets. No holdout or grouped-method claim.'))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.perf_counter()
    print('START',a.mode,n,'single-thread; detailed progress in',out/'launcher.log',flush=True)
    with (out/'launcher.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=a.timeout,env=env)
    result=json.loads((out/'upstream.json').read_text())
    if not result.get('results'):raise ValueError('No upstream result rows')
    rows=result['results'];frontier=[]
    for cell in sorted({(r['selectivity'],r['alpha'],r['mode']) for r in rows}):
        rs=[r for r in rows if (r['selectivity'],r['alpha'],r['mode'])==cell]
        if any(r['num_queries']!=nq or not 0<=r['recall']<=1 or not np.isfinite(r['latency_ms']) or r['latency_ms']<0 for r in rs):raise ValueError('Invalid upstream metrics')
        valid=[r for r in rs if r['recall']>=.95]
        frontier.append(dict(selectivity=cell[0],alpha=cell[1],method=cell[2],
            best_at_recall_95=min(valid,key=lambda r:r['latency_ms']) if valid else None))
    write_json(out/'summary.json',dict(status='complete',wall_seconds=time.perf_counter()-start,
        upstream_result_rows=len(result['results']),frontier=frontier,scope=scope))
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
