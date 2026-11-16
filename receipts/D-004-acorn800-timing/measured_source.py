#!/usr/bin/env python3
"""Pinned upstream ACORN in a separate process; timed search + final group selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diploma_starters.frontier_screen import load_workload, aggregate
from diploma_starters.group_screen import NativeExact,digest,eligibility,evaluate,write_bin,write_json

REV='c259f11cdbec880671658eaa0d8747e2cc9de79d'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload',required=True);p.add_argument('--out',required=True)
    p.add_argument('--upstream',default='external/ACORN');p.add_argument('--binary',default='.local/d004-native/acorn_probe')
    p.add_argument('--split',choices=['tune','test'],default='tune');p.add_argument('--k',type=int,default=100)
    p.add_argument('--pools',nargs='+',type=int,default=[200,800,3200]);p.add_argument('--gamma',type=int,default=1)
    p.add_argument('--efs',nargs='+',type=int,default=[512]);p.add_argument('--index-seed',type=int,default=11)
    p.add_argument('--selectivities',nargs='+',type=float,default=[.1,.01]);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    data,manifest=load_workload(a.workload,a.split);x,q,g=data['vectors'],data['queries'],data['groups']
    if a.k<1 or min(a.pools)<a.k or max(a.pools)>len(x) or a.gamma<1 or min(a.efs)<1 or a.repeats<1 or not all(0<f<=1 for f in a.selectivities):
        p.error('Invalid budgets/selectivities')
    up=Path(a.upstream).resolve();binary=Path(a.binary).resolve()
    if subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()!=REV: p.error('Wrong upstream revision')
    original=subprocess.check_output(['git','-C',str(up),'show','HEAD:CMakeLists.txt'],text=True)
    fixed=original.replace('target_include_directories(faiss PUBLIC ${nlohmann_json_SOURCE_DIR}/include)',
        'target_include_directories(faiss PUBLIC $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>)')
    changed=subprocess.check_output(['git','-C',str(up),'diff','--name-only','HEAD'],text=True).splitlines()
    if any(f!='CMakeLists.txt' for f in changed) or (up/'CMakeLists.txt').read_text() not in [original,fixed]:p.error('Unexpected upstream source modifications')
    build=json.loads(binary.with_suffix('.build.json').read_text())
    adapter=Path(__file__).resolve().parents[1]/'native/acorn_probe.cpp'
    if (build['binary_sha256']!=digest(binary) or build['adapter_sha256']!=digest(adapter)
            or build['library_sha256']!=digest(up/'build/faiss/libfaiss.so') or build['upstream_revision']!=REV):
        p.error('Stale ACORN binary/library: rerun setup_acorn.sh')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);write_json(out/'config.json',vars(a))
    write_bin(out/'query.bin',q);g.astype('<i8').tofile(out/'groups.i64')
    # Predicate labels: rank buckets, not diversity groups. Search uses the exact bitmap.
    (data['ranks']*100//len(x)).astype('<i4').tofile(out/'metadata.i32')
    exact=NativeExact(x,g,'.local/d004-native');rows=[];selected=[];commands=[]
    write_json(out/'manifest.json',dict(workload_sha256=manifest['workload_sha256'],upstream_revision=REV,
        source_sha256=digest(__file__),adapter_sha256=digest(Path(__file__).resolve().parents[1]/'native/acorn_probe.cpp'),
        binary_sha256=digest(binary),upstream_library_sha256=digest(up/'build/faiss/libfaiss.so'),
        cmake_diff=subprocess.check_output(['git','-C',str(up),'diff','HEAD','--','CMakeLists.txt'],text=True),
        timing='Single thread, warmup + repeats. Per-query median includes upstream search and exact rerank/group selection; excludes build, loading and IPC.',
        scope='Unmodified upstream search/construction; adapted SIFT/synthetic metadata. Gamma=1 is not the whole ACORN family.'))
    for fraction in a.selectivities:
        masks=eligibility(data,fraction);maskpath=out/'masks.u8';masks.astype('u1').tofile(maskpath)
        for pool in a.pools:
            for ef in a.efs:
                name=f'acorn{a.gamma}_pool{pool}_ef{ef}';condition=f'both-{fraction:g}'
                result=out/f'{condition}-{name}.bin';log=out/f'{condition}-{name}.log'
                cmd=[str(binary),str(Path(a.workload).resolve()/'base.bin'),str(out/'query.bin'),str(maskpath),
                     str(out/'groups.i64'),str(out/'metadata.i32'),str(result),str(a.k),str(pool),str(ef),str(a.gamma),str(a.index_seed),str(a.repeats)]
                print('START',condition,name,flush=True)
                env=dict(os.environ);env['LD_LIBRARY_PATH']=str(up/'build/faiss')+':'+env.get('LD_LIBRARY_PATH','')
                with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=a.timeout,env=env)
                commands.append(dict(argv=cmd,log=log.name));write_json(out/f'{condition}-{name}.command.json',commands[-1])
                with result.open('rb') as f:
                    shape=np.fromfile(f,'<u4',2)
                    if tuple(shape)!=(len(q),a.k):raise ValueError('Wrong ACORN result shape')
                    ids=np.fromfile(f,'<i8',len(q)*a.k).reshape(len(q),a.k);times=np.fromfile(f,'<f8',len(q))
                    rejected=np.fromfile(f,'<i8',len(q))
                    if len(times)!=len(q) or len(rejected)!=len(q) or f.read(1):raise ValueError('Wrong ACORN result length')
                for qi in range(len(q)):
                    got=ids[qi][ids[qi]!=-1];ref=exact.search(q[qi],masks[qi],a.k,True)
                    rows.append(dict(condition=condition,method=name,query=qi,source_query=int(data['query_ids'][qi]),split=a.split,
                                     ms=float(times[qi]),upstream_ineligible_candidates=int(rejected[qi]),**evaluate(x,q[qi],g,masks[qi],got,ref,True)))
                    selected.append(dict(condition=condition,method=name,query=qi,ids=got.tolist(),reference=ref.tolist()))
                write_json(out/f'{condition}-{name}.metrics.json',[r for r in rows if r['condition']==condition and r['method']==name])
    maskpath.unlink();write_json(out/'per_query.json',rows);write_json(out/'selections.json',selected)
    write_json(out/'summary.json',aggregate(rows));print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
