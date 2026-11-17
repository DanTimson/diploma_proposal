#!/usr/bin/env python3
"""Paired kernel check. Synthetic timings are not an ANN crossover experiment."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diploma_starters.group_screen import NativeExact,write_json,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=100000);p.add_argument('--d',type=int,default=128)
    p.add_argument('--queries',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args()
    if min(a.n,a.d,a.queries)<1 or a.n*a.d>500_000_000:p.error('Invalid or excessive shape')
    rng=np.random.default_rng(20261116);x=rng.normal(size=(a.n,a.d)).astype('f4');q=rng.normal(size=(a.queries,a.d)).astype('f4')
    rows=[];metadata={}
    for layout,g in [('unique',np.arange(a.n)),('skewed',rng.integers(0,min(a.n,1000),a.n))]:
        if layout=='skewed':g[rng.random(a.n)<.8]=0
        engines={key:NativeExact(x,g,'.local/d005-native',implementation=key) for key in ['dense','sparse']}
        metadata={key:e.metadata for key,e in engines.items()}
        for fraction in [.01,.1,.5]:
            for qi in range(a.queries):
                mask=rng.random(a.n)<fraction;times={key:[] for key in engines};got={}
                for rep in range(4):
                    for key in rng.permutation(list(engines)):
                        t=time.perf_counter();got[key]=engines[key].search(q[qi],mask,100,True)
                        if rep:times[key].append((time.perf_counter()-t)*1000)
                np.testing.assert_array_equal(got['dense'],got['sparse'])
                rows.append(dict(layout=layout,fraction=fraction,query=qi,eligible=int(mask.sum()),ids=got['dense'].tolist(),
                                 ms={key:float(np.median(v)) for key,v in times.items()}))
    write_json(a.out,dict(config=vars(a),source_sha256=digest(__file__),backends=metadata,rows=rows,
        note='Same float64 objective; mask enumeration included. Hash aggregation avoids N-sized scratch, but is not assumed faster in all regimes.'))
    print('COMPLETE',a.out)
if __name__=='__main__':main()
