#!/usr/bin/env python3
"""Run pinned upstream build/search, independently evaluate exported candidates.

Filtering here is POST-filtering, not a modification of upstream traversal.
Latencies reported by the upstream CLI exclude our downstream selection.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diploma_starters.group_screen import (NativeExact,conditions,digest,eligibility,evaluate,write_json)

REV='659041e1b12e3ab41c70ad12d11a29a883f17e15'

def read_ids(path,nq,k):
    with Path(path).open('rb') as f:
        shape=np.fromfile(f,dtype='<u4',count=2);ids=np.fromfile(f,dtype='<u4')
    if tuple(shape)!=(nq,k) or len(ids)!=nq*k:raise ValueError('Bad upstream result dimensions')
    return ids.reshape(nq,k).astype('i8')

def choose(x,q,groups,mask,ids,k,grouped):
    # UINT_MAX is upstream's documented missing-result padding, not item zero.
    ids=np.unique(ids[(ids>=0)&(ids<len(x))]);ids=ids[mask[ids]]
    ds=np.square(x[ids].astype('f8')-q.astype('f8')).sum(axis=1)
    order=np.lexsort((ids,ds));out=[];seen=set()
    for i in ids[order]:
        g=int(groups[i])
        if grouped and g in seen:continue
        out.append(int(i));seen.add(g)
        if len(out)==k:break
    return np.array(out,dtype='i8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--out',required=True);p.add_argument('--lists',nargs='+',type=int,default=[200,800,3200])
    p.add_argument('--build-threads',type=int,default=1);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--num-diverse',type=int,default=1,help='Upstream construction blocker threshold; not a final output cap')
    p.add_argument('--cache-dir',default='.local/d003-native');a=p.parse_args()
    run=Path(a.run_dir).resolve();up=Path(a.upstream).resolve();out=Path(a.out).resolve()
    if subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()!=REV:p.error('Wrong upstream revision')
    if subprocess.check_output(['git','-C',str(up),'diff','HEAD','--'],text=True).strip():p.error('Upstream tracked sources modified')
    for name in ['build_memory_index','search_memory_index']:
        if not (up/'build/apps'/name).is_file():p.error(f'Missing upstream binary {name}; build first')
    config=json.loads((run/'config.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if digest(run/'workload.npz')!=manifest['workload_sha256']:p.error('Workload hash mismatch')
    data=dict(np.load(run/'workload.npz',allow_pickle=False));x,q,g=data['vectors'],data['queries'],data['groups'];k=config['k']
    if a.build_threads<1 or a.timeout<1 or a.num_diverse<1 or any(v<k or v>len(x) for v in a.lists):p.error('Bad thread count/budget/list size')
    out.mkdir(parents=True,exist_ok=False);commands=[];rows=[];raw=[]
    def execute(argv,name):
        t=time.perf_counter()
        with (out/f'{name}.log').open('x') as f:
            subprocess.run([str(v) for v in argv],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=a.timeout)
        commands.append(dict(argv=[str(v) for v in argv],seconds=time.perf_counter()-t,log=name+'.log'))
        write_json(out/f'{name}.command.json',commands[-1])
    native=NativeExact(x,g,a.cache_dir)
    for build in ['standard','diverse']:
        prefix=out/build
        argv=[up/'build/apps/build_memory_index','--data_type','float','--dist_fn','l2',
              '--data_path',run/'base.bin','--index_path_prefix',prefix,'-R','64','-L','200','--alpha','1.2','-T',str(a.build_threads)]
        if build=='diverse':argv+=['--seller_file',run/'groups.txt','--NumDiverse',str(a.num_diverse)]
        print('BUILD',build,flush=True);execute(argv,'build-'+build)
        if build=='standard':shutil.copyfile(run/'groups.txt',str(prefix)+'_sellers.txt')
        for search,per_group in ([('ordinary',0),('diverse',1),('diverse',4)] if build=='standard' else [('diverse',1),('diverse',4)]):
            for L in a.lists:
                name=f'{build}-{search}-cap{per_group}-L{L}';print('SEARCH',name,flush=True)
                result=out/name
                argv=[up/'build/apps/search_memory_index','--data_type','float','--dist_fn','l2',
                      '--index_path_prefix',prefix,'--query_file',run/'query.bin','--result_path',result,
                      '-K',str(L),'-L',str(L),'-T','1','--diverse_search',str(search=='diverse').lower(),
                      '--max_K_per_seller',str(per_group),'--post_process','false']
                execute(argv,name)
                ids=read_ids(str(result)+f'_{L}_idx_uint32.bin',len(q),L)
                for cond,grouped,fraction in conditions(config['selectivities']):
                    if not grouped and search=='diverse':continue
                    masks=eligibility(data,fraction)
                    for qi in range(len(q)):
                        # Invalid non-padding IDs cause failure, not silent removal.
                        bad=ids[qi][(ids[qi]>=len(x))&(ids[qi]!=2**32-1)]
                        if len(bad):raise ValueError('Upstream returned out-of-range IDs')
                        got=choose(x,q[qi],g,masks[qi],ids[qi],k,grouped)
                        ref=native.search(q[qi],masks[qi],k,grouped)
                        row=dict(condition=cond,query=qi,method=name,split='tuning' if qi<len(q)//2 else 'confirmation',
                                 **evaluate(x,q[qi],g,masks[qi],got,ref,grouped))
                        rows.append(row);raw.append(dict(condition=cond,query=qi,method=name,ids=got.tolist(),reference=ref.tolist()))
                write_json(out/f'{name}.metrics.json',[r for r in rows if r['method']==name])
    summary=[]
    for key in sorted({(r['condition'],r['method'],r['split']) for r in rows}):
        rs=[r for r in rows if (r['condition'],r['method'],r['split'])==key];entry=dict(zip(['condition','method','split'],key))
        for field in ['item_recall','group_recall','fill','mean_l2_regret','distance_ratio']:
            vs=[r[field] for r in rs if r[field] is not None];entry[field]=float(np.mean(vs)) if vs else None
        entry['violations']=sum(sum(r[k] for k in ['eligibility_violations','group_violations','invalid_ids','duplicate_ids']) for r in rs)
        summary.append(entry)
    write_json(out/'summary.json',summary);write_json(out/'selections.json',raw)
    write_json(out/'manifest.json',dict(upstream_revision=REV,source_sha256=digest(__file__),commands=commands,
        workload_sha256=manifest['workload_sha256'],scope='Unmodified upstream algorithm on adapted workload; post-filter composition',
        timing='Upstream log mean latency is search-only; no end-to-end speed comparison',
        binaries={n:digest(up/'build/apps'/n) for n in ['build_memory_index','search_memory_index']}))
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
