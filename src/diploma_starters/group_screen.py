"""D-003 factorial screen. Adapted benchmarks, never a numerical paper reproduction."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from .ann import Workload, HNSW, Flat
from .assets import fetch
ROOT=Path(__file__).resolve().parents[2]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)

def fvecs(path):
    raw=np.memmap(path,dtype='<i4',mode='r');d=int(raw[0])
    if d<=0 or len(raw)%(d+1):raise ValueError('Invalid fvecs dimensions')
    rows=raw.reshape(-1,d+1)
    if not np.all(rows[:,0]==d):raise ValueError('Variable fvecs dimensions')
    return rows[:,1:].view('<f4')

def write_bin(path,x):
    x=np.asarray(x,dtype='<f4')
    with Path(path).open('wb') as f:
        np.array(x.shape,dtype='<u4').tofile(f);x.tofile(f)

def make_data(kind,n,nq,seed,data_dir):
    rng=np.random.default_rng(seed)
    if kind=='sift':
        lock=json.loads((ROOT/'topics/D_hnsw/d003/sources.json').read_text())['sift']
        for name,spec in lock.items():fetch(spec,Path(data_dir)/name)
        base=fvecs(Path(data_dir)/'sift_base.fvecs');query=fvecs(Path(data_dir)/'sift_query.fvecs')
        if n>len(base) or nq>len(query):raise ValueError('Sample exceeds dataset')
        bids=np.sort(rng.choice(len(base),n,replace=False));qids=np.sort(rng.choice(len(query),nq,replace=False))
        x=np.ascontiguousarray(base[bids]);q=np.ascontiguousarray(query[qids])
    else:
        centers=rng.normal(size=(64,8));proj=np.linalg.qr(rng.normal(size=(64,8)))[0].T
        x=((centers[rng.integers(64,size=n)]+.15*rng.normal(size=(n,8)))@proj).astype('f4')
        q=((centers[rng.integers(64,size=nq)]+.15*rng.normal(size=(nq,8)))@proj).astype('f4')
        bids=np.arange(n);qids=np.arange(nq)
    # Distribution described by ICML SIFT-skewed, newly seeded assignment.
    ng=min(1000,max(20,n//4));groups=rng.integers(1,ng,size=n,dtype=np.int64)
    groups[rng.random(n)<.8]=0
    ranks=rng.permutation(n);buckets=rng.integers(0,100,size=nq)
    return dict(vectors=x,queries=q,groups=groups,ranks=ranks,buckets=buckets,base_ids=bids,query_ids=qids)

class NativeExact:
    def __init__(self,x,groups,cache,implementation='dense'):
        if implementation not in ('dense','sparse'):raise ValueError('Unknown exact implementation')
        x=np.ascontiguousarray(x,dtype='f4');groups=np.ascontiguousarray(groups,dtype='i8')
        if x.ndim!=2 or not all(x.shape) or groups.shape!=(len(x),) or not np.isfinite(x).all():
            raise ValueError('Invalid exact-search vectors/groups')
        if np.any(groups<0) or np.any(groups>=len(x)):raise ValueError('Group ID out of range')
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        source=ROOT/('native/exact_groups_sparse.cpp' if implementation=='sparse' else 'native/exact_groups.cpp');compiler=os.environ.get('CXX','g++')
        version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
        key=hashlib.sha256((digest(source)+version+platform.machine()).encode()).hexdigest()[:20]
        lib=cache/f'exact-{key}.so'
        if not lib.exists():
            tmp=cache/f'exact-{key}-{os.getpid()}.so'
            try:
                subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
                tmp.replace(lib)
            finally:tmp.unlink(missing_ok=True)
        self.lib=ctypes.CDLL(str(lib));self.fn=self.lib.exact_groups;ptr=np.ctypeslib.ndpointer
        self.fn.argtypes=[ptr(np.float32,flags='C_CONTIGUOUS'),ptr(np.float32,flags='C_CONTIGUOUS'),
            ptr(np.int64,flags='C_CONTIGUOUS'),ptr(np.int64,flags='C_CONTIGUOUS'),ctypes.c_int64,
            ctypes.c_int64,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr(np.int64,flags='C_CONTIGUOUS')]
        self.fn.restype=ctypes.c_int
        self.x=np.ascontiguousarray(x,dtype='f4');self.groups=np.ascontiguousarray(groups,dtype='i8')
        self.metadata=dict(implementation=implementation,compiler=version,flags=['-O3','-std=c++17','-shared','-fPIC'],
            source_sha256=digest(source),binary_sha256=digest(lib),accumulation='float64')
    def search(self,q,mask,k,grouped):
        q=np.ascontiguousarray(q,dtype='f4');mask=np.asarray(mask,dtype=bool)
        if q.shape!=(self.x.shape[1],) or mask.shape!=(len(self.x),) or not np.isfinite(q).all() or k<1:
            raise ValueError('Invalid exact-search query/mask/k')
        ids=np.flatnonzero(mask).astype('i8');out=np.empty(k,dtype='i8')
        count=self.fn(self.x,np.ascontiguousarray(q,dtype='f4'),ids,self.groups,len(ids),len(self.x),
                      self.x.shape[1],k,int(grouped),out)
        if count<0:raise ValueError(f'Native exact input error {count}')
        return out[:count]

def eligibility(data,fraction):
    n=len(data['vectors'])
    return np.array([((data['ranks']+int(b)*n//100)%n)<max(1,int(n*fraction)) for b in data['buckets']])

def evaluate(x,q,groups,mask,got,ref,grouped):
    got=np.asarray(got,dtype='i8');ref=np.asarray(ref,dtype='i8')
    valid=(got>=0)&(got<len(x));invalid=int((~valid).sum());got=got[valid]
    dup=len(got)-len(np.unique(got));space=groups if grouped else np.arange(len(x))
    target=set(space[ref].tolist());found=set(space[got].tolist())
    dist=lambda ii:np.square(x[ii].astype('f8')-q.astype('f8')).sum(axis=1)
    a=dist(ref).sum();b=dist(got).sum()
    return dict(item_recall=len(set(got.tolist())&set(ref.tolist()))/len(ref) if len(ref) else None,
        group_recall=len(found&target)/len(target) if target else None,fill=len(got)/len(ref) if len(ref) else None,
        eligibility_violations=int((~mask[got]).sum()),group_violations=len(got)-len(found) if grouped else 0,
        invalid_ids=invalid,duplicate_ids=dup,
        mean_l2_regret=float((b-a)/len(ref)) if len(got)==len(ref) and len(ref) else None,
        distance_ratio=float(b/a) if len(got)==len(ref) and a>0 else None)

def conditions(selectivities):
    return [('neither',False,1.),('group',True,1.)]+[(f'{kind}-{f:g}',g,f) for f in selectivities for kind,g in [('filter',False),('both',True)]]

def summarize(rows):
    result=[]
    for key in sorted({(r['condition'],r['method'],r['ef'],r['split']) for r in rows}):
        rs=[r for r in rows if (r['condition'],r['method'],r['ef'],r['split'])==key]
        entry=dict(zip(['condition','method','ef','split'],key));entry['queries']=len(rs)
        for field in ['item_recall','group_recall','fill','mean_l2_regret','distance_ratio']:
            v=[r[field] for r in rs if r[field] is not None];entry[field]=float(np.mean(v)) if v else None
        entry['p50_ms']=float(np.median([r['ms'] for r in rs]));entry['p95_ms']=float(np.quantile([r['ms'] for r in rs],.95))
        entry['violations']=sum(sum(r[k] for k in ['eligibility_violations','group_violations','invalid_ids','duplicate_ids']) for r in rs)
        result.append(entry)
    return result

def run(a):
    import faiss
    from threadpoolctl import threadpool_limits
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);write_json(out/'config.json',vars(a))
    data=make_data(a.dataset,a.n,a.queries,a.data_seed,a.data_dir)
    x,q,g=data['vectors'],data['queries'],data['groups'];np.savez(out/'workload.npz',**data)
    write_bin(out/'base.bin',x);write_bin(out/'query.bin',q);np.savetxt(out/'groups.txt',g,fmt='%d')
    w=Workload('d003',x,q,g,np.ones((len(q),len(x)),bool),'l2').validate()
    with threadpool_limits(limits=1):
        native=NativeExact(x,g,a.cache_dir);idx=HNSW(w,m=32,ef_construction=200,seed=a.index_seed);flat=Flat(w)
        write_json(out/'manifest.json',dict(schema=1,config=vars(a),numpy=np.__version__,faiss=faiss.__version__,
            python=platform.python_version(),machine=platform.machine(),native=native.metadata,
            workload_sha256=digest(out/'workload.npz'),source_sha256=digest(__file__),
            hnsw_build_seconds=idx.build_seconds,hnsw_bytes=idx.index_bytes,
            timing='one warmup + repeats; single CPU thread; independent searches',
            scope='Adapted factorial screen; not numerical paper reproduction',
            query_split='first half tuning; second half confirmation; no automatic selection'))
        rows=[];selections=[];order=np.random.default_rng(113)
        methods=[('exact_cpp',0),('exact_faiss',0)]+[(f'hnsw_{kind}',ef) for ef in a.efs for kind in ['post','native']]
        for cond,grouped,f in conditions(a.selectivities):
            print('START',cond,flush=True);w.groups=g if grouped else np.arange(len(x),dtype='i8');w.allowed=eligibility(data,f)
            for qi in range(len(q)):
                ref=native.search(q[qi],w.allowed[qi],a.k,grouped);samples={m:[] for m in methods}
                for rep in range(a.repeats+1):
                    for name,ef in [methods[i] for i in order.permutation(len(methods))]:
                        t=time.perf_counter()
                        if name=='exact_cpp':got=native.search(q[qi],w.allowed[qi],a.k,grouped)
                        elif name=='exact_faiss':got,_=flat.search(qi,a.k,0,min(a.pool,len(x)),len(x),adaptive=True,filtered=True)
                        else:got,_=idx.search(qi,a.k,ef,min(a.pool,len(x)),min(a.cap,len(x)),adaptive=True,filtered=name=='hnsw_native')
                        elapsed=(time.perf_counter()-t)*1000
                        if rep:samples[(name,ef)].append(elapsed)
                        if rep==a.repeats:
                            row=dict(condition=cond,query=qi,source_query=int(data['query_ids'][qi]),
                                split='tuning' if qi<len(q)//2 else 'confirmation',method=name,ef=ef,
                                eligible=int(w.allowed[qi].sum()),eligible_groups=len(np.unique(g[w.allowed[qi]])),
                                **evaluate(x,q[qi],g,w.allowed[qi],got,ref,grouped))
                            row['ms']=float(np.median(samples[(name,ef)]));rows.append(row)
                            selections.append(dict(condition=cond,query=qi,method=name,ef=ef,ids=got.tolist(),reference=ref.tolist()))
            # Checkpoint each completed condition, without replacing earlier receipts.
            write_json(out/f'checkpoint-{cond}.json',[r for r in rows if r['condition']==cond])
        write_json(out/'per_query.json',rows);write_json(out/'selections.json',selections)
        write_json(out/'summary.json',summarize(rows));print('COMPLETE',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['clustered','sift'],default='clustered')
    p.add_argument('--n',type=int,default=10000);p.add_argument('--queries',type=int,default=32)
    p.add_argument('--k',type=int,default=10);p.add_argument('--data-seed',type=int,default=20261115)
    p.add_argument('--index-seed',type=int,default=11);p.add_argument('--efs',nargs='+',type=int,default=[128,512])
    p.add_argument('--pool',type=int,default=200);p.add_argument('--cap',type=int,default=5000)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--selectivities',nargs='+',type=float,default=[.5,.1,.01])
    p.add_argument('--data-dir',default='external/sift1m');p.add_argument('--cache-dir',default='.local/d003-native');p.add_argument('--out',required=True)
    a=p.parse_args()
    if not(a.n>=100 and 2<=a.queries<=10000 and a.n*a.queries<=300_000_000 and 0<a.k<=a.n and a.repeats>=1 and a.pool>0 and
           a.cap>=a.pool and all(e>0 for e in a.efs) and all(0<s<1 for s in a.selectivities)):
        p.error('Invalid dimensions, budgets, or selectivities')
    run(a)
if __name__=='__main__':main()
