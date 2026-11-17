"""D-008 final bounded development screen; see topics/D_hnsw/d008/DECISION.md."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import numpy as np
from .explore import (ROOT, Kernel, atomic, identity, source_receipt, conditions,
                      select_candidates, replay)
from .group_screen import NativeExact, digest, evaluate
from .upstream_controls import build_probe, prepare_candidates
from .retention import RetentionKernel, FastSelector

GRID = [(256,256),(1024,256),(1024,1024),(4096,256),(4096,1024)]
UPSTREAM_GRID = [(256,256),(1024,256),(1024,1024),(4096,1024)]


class SIMDExact(NativeExact):
    """Same complete scan/selection, with explicit double-precision SIMD reduction."""
    def __init__(self,x,groups,cache):
        # Reuse the established input validation and ABI definition.
        super().__init__(x,groups,cache)
        source=ROOT/'native/exact_groups_simd.cpp'
        compiler=os.environ.get('CXX','g++')
        version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
        flags=['-O3','-std=c++17','-shared','-fPIC','-fopenmp-simd','-march=native']
        key=identity([digest(source),version,flags,platform.machine()])[:20]
        cache=Path(cache);binary=cache/f'exact-simd-{key}.so'
        if not binary.exists():
            tmp=cache/f'exact-simd-{key}-{os.getpid()}.so'
            try:
                subprocess.run([compiler,*flags,str(source),'-o',str(tmp)],check=True)
                tmp.replace(binary)
            finally:tmp.unlink(missing_ok=True)
        argtypes=self.fn.argtypes
        self.lib=ctypes.CDLL(str(binary));self.fn=self.lib.exact_groups_simd
        self.fn.argtypes=argtypes;self.fn.restype=ctypes.c_int
        self.metadata=dict(compiler=version,flags=flags,source_sha256=digest(source),
                           binary_sha256=digest(binary),accumulation='float64 SIMD reduction')


def upper_entry(index,q):
    """FAISS 1.12 greedy upper-level descent; counts actual distance calls, not unique IDs."""
    import faiss
    q=np.ascontiguousarray(q,dtype='f4')
    if q.shape!=(index.d,) or not np.isfinite(q).all() or index.ntotal<1:
        raise ValueError('Invalid hierarchical-entry query/index')
    h=index.hnsw;dc=index.storage.get_distance_computer();dc.set_query(faiss.swig_ptr(q))
    node=np.array([h.entry_point],dtype='i4');distance=np.array([dc(int(node[0]))],dtype='f4')
    calls=1;hops=0
    for level in range(h.max_level,0,-1):
        stats=faiss.greedy_update_nearest(h,dc,level,faiss.swig_ptr(node),faiss.swig_ptr(distance))
        calls+=int(stats.ndis);hops+=int(stats.nhops)
    return int(node[0]),dict(upper_distances=calls,upper_hops=hops)


def diagnostic(kernel,index,q,groups,mask,starts,k,budget,entry):
    stats={};upper_ms=0.
    if entry=='hierarchical':
        t=time.perf_counter();node,stats=upper_entry(index,q);upper_ms=(time.perf_counter()-t)*1000
        starts=[node]
    spent=stats.get('upper_distances',0)
    if budget-spent<len(starts):
        raise ValueError('Upper descent consumes the diagnostic budget; choose a larger budget')
    got,lower=kernel.search(q,groups,mask,starts,budget-spent,128,0,0,k)
    return got,dict(**lower,**stats,total_distances=spent+lower['distances'],upper_ms=upper_ms)


def faiss_candidates(index,q,mask,ef,pool,filtered):
    """No max(ef,pool): parameters are independently passed and explicitly reported."""
    import faiss
    if min(ef,pool)<1:raise ValueError('Positive ef/pool required')
    params=faiss.SearchParametersHNSW();params.efSearch=ef
    t=time.perf_counter()
    if filtered:
        bitmap=np.packbits(mask,bitorder='little')
        params.sel=faiss.IDSelectorBitmap(len(mask),faiss.swig_ptr(bitmap))
    packing_ms=(time.perf_counter()-t)*1000
    faiss.cvar.hnsw_stats.reset()
    t=time.perf_counter();_,ids=index.search(q[None,:],pool,params=params)
    search_ms=(time.perf_counter()-t)*1000
    return ids[0],dict(ef_search=ef,pool=pool,mask_ms=packing_ms,search_ms=search_ms,
        search_distances=int(faiss.cvar.hnsw_stats.ndis),
        returned_candidates=int(np.sum(ids[0]>=0)))


def numerical_audit(x,q,got,reference):
    if np.array_equal(got,reference):return 'identical'
    dist=lambda ids:np.sort(np.square(x[ids].astype('f8')-q.astype('f8')).sum(axis=1))
    if len(got)!=len(reference) or not np.allclose(dist(got),dist(reference),rtol=1e-10,atol=1e-10):
        raise ValueError('SIMD exhaustive selection disagrees beyond numerical tolerance')
    return 'different_ids_with_near_equal_distances'


def summarize(rows):
    table=[]
    for key in sorted({(r['condition'],r['k'],r['method']) for r in rows}):
        rs=[r for r in rows if (r['condition'],r['k'],r['method'])==key]
        out=dict(zip(('condition','k','method'),key));out['queries']=len(rs)
        for field in ('item_recall','group_recall','fill','distance_ratio','mean_l2_regret','ms'):
            vs=[r[field] for r in rs if r[field] is not None]
            out[field]=float(np.mean(vs)) if vs else None
        out['queries_at_recall95']=sum(r['item_recall'] is not None and r['item_recall']>=.95 for r in rs)
        out['minimum_item_recall']=min((r['item_recall'] for r in rs if r['item_recall'] is not None),default=None)
        out['feasible_queries']=sum(r['feasible'] for r in rs)
        out['violations']=sum(r[f] for r in rs for f in ('eligibility_violations','group_violations','invalid_ids','duplicate_ids'))
        out['timing_scope']=rs[0]['timing_scope'];table.append(out)
    return table


def make_summary(out,rows,status,elapsed):
    table=summarize(rows)
    atomic(out/'summary.json',dict(status=status,wall_seconds_this_invocation=elapsed,table=table))
    lines=['# D-008 final bounded development screen','',
        'Upstream ms is separately measured search + downstream selection, not integrated end-to-end latency.',
        'Held-out queries were not used. Parameters and distance counts are not interchangeable across methods.','',
        '| Condition | k | Method | Item recall | Group recall | Fill | Mean ms | Queries >=95% |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'NA' if v is None else f'{v:.4f}'
    for r in table:
        lines.append(f"| {r['condition']} | {r['k']} | {r['method']} | {fmt(r['item_recall'])} | "
                     f"{fmt(r['group_recall'])} | {fmt(r['fill'])} | {r['ms']:.3f} | {r['queries_at_recall95']}/{r['queries']} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def run(a):
    import faiss
    from threadpoolctl import threadpool_limits
    invocation_started=time.perf_counter()
    prepared=Path(a.prepared).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((prepared/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(prepared/name)!=sha:raise ValueError('Prepared file changed: '+name)
    if digest(prepared/'hnsw0.npy')!=manifest['graphs']['hnsw0']['graph_sha256']:
        raise ValueError('Prepared level-zero graph changed')
    if faiss.__version__!='1.12.0':raise ValueError('This control requires FAISS 1.12.0')
    data=dict(np.load(prepared/'workload.npz'));nq=a.queries
    if not 1<=nq<=len(data['queries']):raise ValueError('Query count exceeds frozen development split')
    grid=[(128,128),(256,128)] if a.smoke else GRID
    ugrid=[(128,128)] if a.smoke else UPSTREAM_GRID
    budgets=[1024] if a.smoke else [1024,4096,16384]
    code=source_receipt()
    for p in [Path(__file__),ROOT/'src/diploma_starters/upstream_controls.py',ROOT/'native/exact_groups_simd.cpp', ROOT/'native/retention.cpp', ROOT/'src/diploma_starters/retention.py', ROOT/'topics/D_hnsw/d008/DECISION.md', ROOT/'src/diploma_starters/d008_decision.py', ROOT/'src/diploma_starters/baseline_controls.py']:
        code[str(p.relative_to(ROOT))]=digest(p)
    lock=out/'RUNNING.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        probe=None;probe_info=None
        if a.upstream:
            for graph in ('diskann','diverse_diskann'):
                receipt=json.loads((prepared/f'{graph}-receipt.json').read_text())
                for name,sha in receipt['files'].items():
                    if digest(prepared/name)!=sha:raise ValueError('Upstream graph checkpoint changed')
            print('CHECK/BUILD upstream logging probe',flush=True)
            probe,probe_info=build_probe(a.upstream,Path(a.cache_dir)/'upstream')
            if probe_info.get('timing_clock')!='std::chrono::steady_clock':
                raise ValueError('D-007a monotonic-clock patch is required')
        plan=dict(schema=1,prepared_manifest=digest(prepared/'manifest.json'),code=code,
            queries=nq,query_ids=data['query_ids'][:nq].tolist(),ks=a.ks,repeats=a.repeats,
            grid=grid,upstream_grid=ugrid if a.upstream else [],budgets=budgets,
            upstream=probe_info,threads=1,smoke=a.smoke)
        # Normalize tuples before comparing JSON plans on resume.
        plan=json.loads(json.dumps(plan))
        plan_path=out/'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
            raise ValueError('D-008 plan differs; use a new output directory')
        atomic(plan_path,plan);atomic(out/'prepared_manifest.json',manifest)
        with threadpool_limits(limits=1):
            faiss.omp_set_num_threads(1)
            candidates=(prepare_candidates(prepared,out,data,nq,a.repeats,probe,ugrid,a.timeout,
                                           Path(a.upstream).resolve()/'build/apps/search_memory_index')
                        if probe is not None else {})
            execute(a,prepared,out,manifest,data,plan,candidates,invocation_started)
    finally:lock.unlink(missing_ok=True)


def execute(a,prepared,out,manifest,data,plan,upstream,started):
    import faiss
    rows=[]
    x=np.load(prepared/'vectors.npy',mmap_mode='r');groups=data['groups']
    index=faiss.read_index(str(prepared/'hnsw.faiss'))
    exact=NativeExact(x,groups,a.cache_dir);simd=SIMDExact(x,groups,a.cache_dir)
    kernel=RetentionKernel(x,np.load(prepared/'hnsw0.npy'),a.cache_dir)
    selector=FastSelector(x,groups,a.cache_dir)
    atomic(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        numpy=np.__version__,faiss=faiss.__version__,exact=exact.metadata,simd=simd.metadata,kernel=kernel.receipt))
    supplied=json.loads((prepared/'published_ground_truth.json').read_text());replays=[]
    plan_hash=identity(plan)
    for qi in range(a.queries):
        q=data['queries'][qi]
        replays.append(replay(x,q,data['counts']==data['labels'][qi],supplied[qi],exact,min(100,len(supplied[qi]))))
        atomic(out/'em_replay.json',replays)
        # Offline composition diagnostic, never provided to a retrieval policy.
        global_best=exact.search(q,np.ones(len(x),bool),len(np.unique(groups)),True)
        best_by_group=np.full(len(x),-1,dtype='i8');best_by_group[groups[global_best]]=global_best
        for condition,mask,grouped in conditions(data,qi,manifest['date_thresholds']):
            if not grouped:continue
            for k in a.ks:
                path=out/'cells'/f'q{qi:04d}-{condition}-k{k}.json'
                if path.exists():
                    saved=json.loads(path.read_text())
                    if saved['plan']!=plan_hash or identity(saved['rows'])!=saved['checksum']:
                        raise ValueError('Completed cell checksum/plan differs')
                    rows.extend(saved['rows']);continue
                reference=exact.search(q,mask,k,True)
                lost=int(np.sum(~mask[best_by_group[groups[reference]]]))
                methods=[('exact_reference',None),('exact_simd',None)]
                for ef,pool in plan['grid']:
                    for mode in ('post','filter'):
                        methods.append((f'faiss_{mode}/ef{ef}/pool{pool}',('faiss',mode,ef,min(pool,len(x)))))
                for budget in plan['budgets']:
                    for entry in ('ordinary_before','ordinary_after','alternate_before','alternate_after'):
                        methods.append((f'hnsw0_{entry}/D{budget}',('graph',entry,budget)))
                for key in upstream:
                    graph,mode,ef,pool=key
                    methods.append((f'upstream_{graph}/{mode}/L{ef}/pool{pool}',('upstream',key)))
                values={};samples={name:[] for name,_ in methods};details={name:[] for name,_ in methods}
                order=np.random.default_rng(8173+qi*101+k)
                for rep in range(a.repeats+1):
                    for pos in order.permutation(len(methods)):
                        name,config=methods[pos];stats={};t=time.perf_counter()
                        if config is None:
                            got=(simd if name=='exact_simd' else exact).search(q,mask,k,True)
                        elif config[0]=='faiss':
                            _,mode,ef,pool=config
                            ids,stats=faiss_candidates(index,q,mask,ef,pool,mode=='filter')
                            ts=time.perf_counter();got=selector.select(q,mask,ids,k)
                            stats['selection_ms']=(time.perf_counter()-ts)*1000
                        elif config[0]=='graph':
                            _,entry,budget=config
                            node,upper=upper_entry(index,q)
                            remaining=budget-upper['upper_distances']
                            if remaining<1:raise ValueError('Upper descent exceeds budget')
                            mode={'ordinary_before':0,'alternate_before':1,'ordinary_after':2,'alternate_after':3}[entry]
                            got,stats=kernel.search(q,groups,mask,[node],remaining,128,mode,0,k)
                            stats.update(upper)
                            stats['total_distances']=upper['upper_distances']+stats['distances']
                        else:
                            ids,search_ms,cmps=upstream[config[1]]
                            got=selector.select(q,mask,ids[qi],k)
                            stats=dict(selection_ms=(time.perf_counter()-t)*1000,
                                search_ms=float(search_ms[max(0,rep-1),qi]),
                                search_distances=int(cmps[max(0,rep-1),qi]),
                                returned_candidates=int(np.sum(ids[qi]>=0)))
                        elapsed=(time.perf_counter()-t)*1000
                        if config and config[0]=='upstream':
                            elapsed=stats['search_ms']+stats['selection_ms']
                        if rep==0 and config and config[0] in ('faiss','upstream'):
                            check_ids=ids if config[0]=='faiss' else ids[qi]
                            numerical_audit(x,q,got,select_candidates(x,q,groups,mask,check_ids,k))
                        if name in values and not np.array_equal(values[name],got):
                            raise ValueError('Repeated search changed results: '+name)
                        values[name]=got
                        if rep:samples[name].append(elapsed);details[name].append(stats)
                cell_rows=[]
                for name,config in methods:
                    got=values[name];metrics=evaluate(x,q,groups,mask,got,reference,True)
                    if any(metrics[f] for f in ('eligibility_violations','group_violations','invalid_ids','duplicate_ids')):
                        raise ValueError('Constraint violation: '+name)
                    audit=numerical_audit(x,q,got,reference) if name=='exact_simd' else None
                    scope='separate_search_plus_selection' if config and config[0]=='upstream' else 'integrated_query'
                    cell_rows.append(dict(condition=condition,k=k,method=name,query_id=int(data['query_ids'][qi]),
                        feasible=len(reference)==k,eligible_items=int(mask.sum()),eligible_groups=len(np.unique(groups[mask])),
                        target_groups_with_ineligible_global_best=lost,reference=reference.tolist(),selected=got.tolist(),
                        ms=float(np.median(samples[name])),samples_ms=samples[name],stage_samples=details[name],
                        timing_scope=scope,numerical_audit=audit,**metrics))
                atomic(path,dict(plan=plan_hash,checksum=identity(cell_rows),rows=cell_rows));rows.extend(cell_rows)
            print(f'COMPLETE query {qi+1}/{a.queries} {condition}',flush=True)
        make_summary(out,rows,'running',time.perf_counter()-started)
    make_summary(out,rows,'complete',time.perf_counter()-started)
    print('COMPLETE',out,flush=True)


def package(out,destination):
    root=Path(out)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_symlink():continue
            if p.is_file() and p.suffix in ('.json','.md','.log','.tsv'):
                z.write(p,str(p.relative_to(root)))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--prepared',required=True);r.add_argument('--out',required=True)
    r.add_argument('--queries',type=int,default=16);r.add_argument('--repeats',type=int,default=2)
    r.add_argument('--ks',type=int,nargs='+',default=[10,100]);r.add_argument('--upstream')
    r.add_argument('--smoke',action='store_true');r.add_argument('--cache-dir',default='.local/d008-native')
    r.add_argument('--timeout',type=int,default=1800)
    pk=sub.add_parser('package');pk.add_argument('--out',required=True);pk.add_argument('--destination',required=True)
    a=p.parse_args()
    if a.command=='run':
        if a.queries<1 or a.repeats<1 or a.timeout<1 or min(a.ks)<1 or len(set(a.ks))!=len(a.ks):
            p.error('Positive queries/repeats/k/timeout and unique k values required')
        run(a)
    else:package(a.out,a.destination)


if __name__=='__main__':main()
