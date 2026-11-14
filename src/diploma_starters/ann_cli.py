"""Topic D executable smoke runner. Heavy sweeps are always explicit."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,help='New JSON result file; never overwritten')
    p.add_argument('--queries',type=int,default=48)
    p.add_argument('--k',type=int,default=10)
    p.add_argument('--efs',type=int,nargs='+',default=[32,128])
    p.add_argument('--pool',type=int,default=40)
    p.add_argument('--cap',type=int,default=1280)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=11)
    sub=p.add_subparsers(dest='case',required=True)
    s=sub.add_parser('synthetic');s.add_argument('--n',type=int,default=6000)
    s.add_argument('--mode',choices=['clean','copies','sparse_filter','adverse_filter'],default='clean')
    s=sub.add_parser('scifact');s.add_argument('directory');s.add_argument('--dimensions',type=int,default=96)
    s=sub.add_parser('recad');s.add_argument('archive');s.add_argument('--dimensions',type=int,default=64)
    s=sub.add_parser('fanns');s.add_argument('directory')
    a=p.parse_args();out=Path(a.out)
    if out.exists():p.error('Output already exists; choose a new run name')
    if a.queries<1:p.error('queries must be positive')
    from . import ann,ann_data
    started=time.perf_counter()
    if a.case=='synthetic':w=ann_data.synthetic(a.n,a.queries,a.seed,a.mode)
    elif a.case=='scifact':w=ann_data.scifact(a.directory,a.queries,a.dimensions,a.seed)
    elif a.case=='recad':w=ann_data.recad(a.archive,a.queries,a.dimensions,a.seed)
    else:w=ann_data.fanns(a.directory,a.queries,a.k)
    preparation_seconds=time.perf_counter()-started
    result=ann.benchmark(w,a.k,a.efs,a.pool,a.cap,a.repeats,a.seed)
    result['preparation_seconds']=preparation_seconds
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'out':str(out),'workload':result['workload'],'summary':result['summary']},indent=2))


if __name__=='__main__':main()
