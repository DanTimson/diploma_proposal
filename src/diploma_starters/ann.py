"""D0/D1 diagnostic: fixed representations, constrained retrieval, no truth inference.

The index is FAISS HNSW, not a new graph-construction algorithm. Group caps
and eligibility are output constraints; ineligible nodes may remain routing nodes.
"""
from dataclasses import dataclass, field
import hashlib
import platform
import time
from pathlib import Path
import numpy as np


@dataclass
class Workload:
    name: str
    vectors: np.ndarray
    queries: np.ndarray
    groups: np.ndarray
    allowed: np.ndarray  # one mask per query; small diagnostics only
    metric: str = 'ip'
    gold: list = field(default_factory=list)  # evaluation only, IDs in groups space
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        self.queries = np.ascontiguousarray(self.queries, dtype=np.float32)
        n, d = self.vectors.shape
        if n == 0 or self.queries.ndim != 2 or self.queries.shape[1] != d or len(self.queries) == 0:
            raise ValueError('Nonempty compatible matrices required')
        if not np.isfinite(self.vectors).all() or not np.isfinite(self.queries).all():
            raise ValueError('Nonfinite vector')
        self.groups = np.asarray(self.groups, dtype=np.int64)
        self.allowed = np.asarray(self.allowed, dtype=bool)
        if self.groups.shape != (n,) or self.allowed.shape != (len(self.queries), n):
            raise ValueError('Group/mask shape mismatch')
        if np.any(self.groups < 0) or self.groups.max() >= n:
            raise ValueError('Groups must be dense nonnegative IDs below database size')
        if self.metric not in ('ip', 'l2'):
            raise ValueError('Metric must be ip or l2')
        if self.gold and len(self.gold) != len(self.queries):
            raise ValueError('Gold length mismatch')
        return self


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norm, 1e-12))


def scores(x, q, metric):
    if metric == 'ip':
        return x @ q
    # Direct subtraction avoids catastrophic cancellation near identical vectors.
    delta = x - q
    return -np.einsum('ij,ij->i', delta, delta)


def select(ids, values, groups, allowed, k):
    """Exact best additive-score set, <=1 item per declared group, cardinality <=k.

    Cardinality has priority, including for negative scores. Not MMR, truth,
    or an optimum for arbitrary provenance/dependence constraints.
    """
    ids = np.asarray(ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    valid = (ids >= 0) & (ids < len(groups))
    ids, values = ids[valid], values[valid]
    ok = allowed[ids]
    ids, values = ids[ok], values[ok]
    if not len(ids):
        return np.empty(0, dtype=np.int64)
    # Select each group's maximum in O(candidates), stable representative at ties.
    _, local_groups = np.unique(groups[ids], return_inverse=True)
    ng = int(local_groups.max()) + 1
    best = np.full(ng, -np.inf, dtype=np.float32)
    np.maximum.at(best, local_groups, values)
    winners = values == best[local_groups]
    representatives = np.full(ng, len(groups), dtype=np.int64)
    np.minimum.at(representatives, local_groups[winners], ids[winners])
    available = np.flatnonzero(np.isfinite(best))
    if len(available) > k:
        threshold = np.partition(best[available], len(available)-k)[len(available)-k]
        # Include all boundary ties before stable truncation.
        available = available[best[available] >= threshold]
    order = np.lexsort((representatives[available], -best[available]))[:k]
    return representatives[available[order]]


def exact(w, qi, k, block_size=2048):
    ids = np.flatnonzero(w.allowed[qi])
    values = np.empty(len(ids), dtype=np.float32)
    for offset in range(0, len(ids), block_size):
        batch=ids[offset:offset+block_size]
        values[offset:offset+block_size]=scores(w.vectors[batch], w.queries[qi], w.metric)
    return select(ids, values, w.groups, w.allowed[qi], k)


class HNSW:
    def __init__(self, w, m=16, ef_construction=100, seed=11):
        import faiss
        self.faiss, self.w = faiss, w
        self.is_flat = False
        faiss.omp_set_num_threads(1)
        metric = faiss.METRIC_INNER_PRODUCT if w.metric == 'ip' else faiss.METRIC_L2
        self.index = faiss.IndexHNSWFlat(w.vectors.shape[1], m, metric)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.rng = faiss.RandomGenerator(seed)
        start = time.perf_counter()
        self.index.add(w.vectors)
        self.build_seconds = time.perf_counter()-start
        self.index_bytes = len(faiss.serialize_index(self.index))

    def search(self, qi, k, ef, pool, cap, adaptive=False, filtered=False):
        w = self.w
        limit = min(len(w.vectors), cap)
        size = min(pool, limit)
        calls, seen = 0, set()
        started = time.perf_counter()
        # This bitmap construction is included in measured query latency.
        # Native selector accepts results; it is NOT an ACORN implementation.
        bitmap = np.packbits(w.allowed[qi], bitorder='little') if filtered else None
        while True:
            params = self.faiss.SearchParameters() if self.is_flat else self.faiss.SearchParametersHNSW()
            if not self.is_flat:
                params.efSearch = max(ef, size)
            if filtered:
                params.sel = self.faiss.IDSelectorBitmap(len(w.vectors), self.faiss.swig_ptr(bitmap))
            _, found = self.index.search(w.queries[qi:qi+1], size, params=params)
            seen.update(int(i) for i in found[0] if i >= 0)
            ids = np.array(sorted(seen), dtype=np.int64)
            selected = select(ids, scores(w.vectors[ids], w.queries[qi], w.metric),
                              w.groups, w.allowed[qi], k)
            calls += 1
            # Having k eligible roots is NOT a certificate of exact top-k.
            exhausted_exact = self.is_flat and np.count_nonzero(found[0] >= 0) < size
            if not adaptive or len(selected) >= k or size >= limit or exhausted_exact:
                break
            size = min(size * 2, limit)
        return selected, dict(ms=(time.perf_counter()-started)*1000,
                              search_calls=calls, unique_candidates=len(seen),
                              final_pool=size, effective_ef=max(ef,size),
                              cap_reached=size >= limit)


class Flat(HNSW):
    """Compiled exhaustive baseline; same candidates/selector/group-cap code."""
    def __init__(self, w):
        import faiss
        self.faiss,self.w,self.is_flat=faiss,w,True
        self.index=(faiss.IndexFlatIP(w.vectors.shape[1]) if w.metric=='ip'
                    else faiss.IndexFlatL2(w.vectors.shape[1]))
        t=time.perf_counter();self.index.add(w.vectors)
        self.build_seconds=time.perf_counter()-t


def collapse_identical_groups(w):
    """Safe only for identical vectors AND identical eligibility within each group.

    This is a preprocessing baseline, not a learned independence detector.
    Return None when any group cannot be represented exactly by a single item.
    """
    _, inverse = np.unique(w.groups,return_inverse=True)
    n_groups=int(inverse.max())+1
    if n_groups==len(w.vectors):return None
    reps=np.full(n_groups,len(w.vectors),dtype=np.int64)
    np.minimum.at(reps,inverse,np.arange(len(w.vectors)))
    if not np.array_equal(w.vectors,w.vectors[reps[inverse]]):return None
    if not np.array_equal(w.allowed,w.allowed[:,reps[inverse]]):return None
    compact=Workload(w.name+'_collapsed',w.vectors[reps],w.queries,np.arange(n_groups),
                     w.allowed[:,reps],w.metric).validate()
    return compact,reps


def evaluate_selection(w, qi, result, reference):
    got = set(map(int, w.groups[result]))
    target = set(map(int, w.groups[reference]))
    denom = len(target)
    gs = w.gold[qi] if w.gold else {}
    out = dict(group_recall=len(got & target)/denom if denom else None,
               fill=len(result)/denom if denom else None,
               eligible=bool(w.allowed[qi,result].all()),
               unique=len(got)==len(result), empty_reference=not denom)
    # Score gap is meaningful only at equal cardinality. Normalized per result,
    # never divide by an IP sum (may be negative or zero).
    out['mean_score_gap'] = (float((scores(w.vectors[reference],w.queries[qi],w.metric).sum()
                            - scores(w.vectors[result],w.queries[qi],w.metric).sum())/denom)
                            if denom and len(result)==denom else None)
    for key, relevant in gs.items():
        relevant = set(relevant)
        out[key+'_recall'] = len(got & relevant)/len(relevant) if relevant else None
    return out


def benchmark(w, k=10, efs=(32,128), pool=40, cap=1280, repeats=3, seed=11, progress=None):
    """Warm per-query latency, single-thread CPU, no timing pass/fail assertions."""
    import faiss
    from threadpoolctl import threadpool_limits
    w.validate()
    if k < 1 or pool < k or cap < pool or repeats < 1 or min(efs) < 1:
        raise ValueError('Require 0<k<=pool<=cap, positive repeats and ef')
    with threadpool_limits(limits=1):
        if progress: progress('building HNSW')
        h = HNSW(w,seed=seed)
        if progress: progress('building exact/collapse controls')
        flat = Flat(w)
        compact=collapse_identical_groups(w)
        compact_h=HNSW(compact[0],seed=seed) if compact else None
        compact_flat=Flat(compact[0]) if compact else None
        if progress: progress('computing exact references')
        refs = [exact(w,i,k) for i in range(len(w.queries))]
        methods = [('exact_prefilter',None),('exact_faiss',None)] + [(kind,ef) for ef in efs for kind in
                    ['hnsw_fixed','hnsw_adaptive','hnsw_filtered']]
        if compact:
            methods += [('exact_collapsed_faiss',None)] + [('hnsw_collapsed',ef) for ef in efs]
        measured = {f'{kind}_ef{ef}' if ef else kind: [] for kind,ef in methods}
        # A fixed measurement RNG is separate from the construction seed.
        # Shuffle per query/repetition so one wrapper isn't always cache-warmed
        # by another wrapper that performs effectively the same search.
        order_rng=np.random.default_rng(20261114)
        for rep in range(repeats+1):
            if progress: progress(f'query repetition {rep}/{repeats} (0=warmup)')
            for qi in range(len(w.queries)):
                order=[methods[i] for i in order_rng.permutation(len(methods))]
                for kind,ef in order:
                    key = f'{kind}_ef{ef}' if ef else kind
                    if kind == 'exact_prefilter':
                        t=time.perf_counter();result=exact(w,qi,k)
                        stats=dict(ms=(time.perf_counter()-t)*1000,search_calls=0,
                                   unique_candidates=int(w.allowed[qi].sum()),cap_reached=False)
                    elif kind == 'exact_faiss':
                        result,stats=flat.search(qi,k,1,pool,len(w.vectors),adaptive=True,filtered=True)
                    elif kind in ('exact_collapsed_faiss','hnsw_collapsed'):
                        engine=compact_flat if kind=='exact_collapsed_faiss' else compact_h
                        result,stats=engine.search(qi,k,ef or 1,pool,len(compact[0].vectors),
                                                   adaptive=True,filtered=True)
                        result=compact[1][result]
                    else:
                        result,stats=h.search(qi,k,ef,pool,cap,
                              adaptive=kind!='hnsw_fixed',filtered=kind=='hnsw_filtered')
                    if rep:
                        measured[key].append(dict(query=qi,repeat=rep,**stats,
                                  **evaluate_selection(w,qi,result,refs[qi])))
        summary={}
        for key,rows in measured.items():
            summary[key]={'p50_ms':float(np.median([r['ms'] for r in rows])),
                          'p95_ms':float(np.percentile([r['ms'] for r in rows],95)),
                          'eligibility_violations':sum(not r['eligible'] for r in rows),
                          'group_violations':sum(not r['unique'] for r in rows)}
            for metric in ['group_recall','fill','mean_score_gap','unique_candidates','search_calls',
                           'evidence_recall','support_recall','contradict_recall','positive_recall']:
                vals=[r[metric] for r in rows if r.get(metric) is not None]
                if vals:summary[key][metric]=float(np.mean(vals))
            summary[key]['empty_reference_queries']=sum(len(r)==0 for r in refs)
        exact_best=min(row['p50_ms'] for key,row in summary.items() if key.startswith('exact_'))
        for key,row in summary.items():
            row['speedup_p50_vs_best_exact']=exact_best/row['p50_ms']
        fingerprint=hashlib.sha256()
        for array in (w.vectors,w.queries,w.groups,w.allowed):
            fingerprint.update(memoryview(np.ascontiguousarray(array)).cast('B'))
        return dict(workload=w.name,n=len(w.vectors),queries=len(w.queries),d=w.vectors.shape[1],
                    metric=w.metric,metadata=w.metadata,seed=seed,k=k,pool=pool,cap=cap,efs=list(efs),
                    repeats=repeats,threads=1,faiss=faiss.__version__,numpy=np.__version__,
                    measurement_order='per-query/per-repetition shuffle; fixed RNG 20261114',
                    python=platform.python_version(),machine=platform.machine(),
                    build_parameters={'M':16,'efConstruction':100,'random_seed':seed},
                    implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ['ann.py','ann_data.py','ann_cli.py']},
                    build_seconds=h.build_seconds,flat_build_seconds=flat.build_seconds,
                    collapsed_index=(dict(n=len(compact[0].vectors),build_seconds=compact_h.build_seconds,
                                     serialized_bytes=compact_h.index_bytes) if compact else None),
                    serialized_index_bytes=h.index_bytes,
                    vector_bytes=w.vectors.nbytes,
                    workload_sha256=fingerprint.hexdigest(),
                    summary=summary,per_query=measured,
                    warning='Smoke only. Serialized bytes are not peak RSS. No paper speedup reproduction.')
