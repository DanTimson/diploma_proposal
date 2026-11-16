"""D-004: frozen data, fresh queries, bounded experiments and resumable controls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .ann import HNSW, Workload
from .group_screen import (NativeExact, digest, eligibility, evaluate, fvecs,
                           make_data, write_bin, write_json)
from .resume_graph import Graph


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def fresh_data(kind, n, old_queries, queries, old_seed, query_seed, data_dir):
    # Replay D-003 exactly, then replace ONLY query-dependent arrays.
    data = make_data(kind, n, old_queries, old_seed, data_dir)
    old_ids = data['query_ids'].copy()
    rng = np.random.default_rng(query_seed)
    if kind == 'sift':
        all_q = fvecs(Path(data_dir)/'sift_query.fvecs')
        available = np.setdiff1d(np.arange(len(all_q)), old_ids)
        if queries > len(available): raise ValueError('Not enough unseen queries')
        ids = rng.choice(available, queries, replace=False)
        q = np.ascontiguousarray(all_q[ids])
    else:
        geometry = np.random.default_rng(old_seed)
        centers = geometry.normal(size=(64, 8))
        proj = np.linalg.qr(geometry.normal(size=(64, 8)))[0].T
        q = ((centers[rng.integers(64, size=queries)]+.15*rng.normal(size=(queries, 8)))@proj).astype('f4')
        ids = np.arange(old_queries, old_queries+queries)
    # Separate mask RNG: increasing query count preserves prefix predicates.
    data.update(queries=q, query_ids=ids, buckets=np.random.default_rng(query_seed+1).integers(0, 100, queries))
    return data, old_ids


def prepare(a):
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    data, old = fresh_data(a.dataset, a.n, a.old_queries, a.queries, a.data_seed, a.query_seed, a.data_dir)
    out.mkdir(parents=True)
    np.savez(out/'workload.npz', **data)
    write_bin(out/'base.bin', data['vectors']); np.savetxt(out/'groups.txt', data['groups'], fmt='%d')
    write_json(out/'config.json', vars(a))
    write_json(out/'manifest.json', dict(schema=1, workload_sha256=digest(out/'workload.npz'),
        array_hashes={key:array_hash(v) for key,v in data.items()},
        old_query_ids=old.tolist(), fresh_query_ids=data['query_ids'].tolist(),
        tune_indices=list(range(a.queries//2)), test_indices=list(range(a.queries//2, a.queries)),
        numpy=np.__version__, source_sha256=digest(__file__),
        scope='Real SIFT vectors with synthetic groups/predicates, or fully synthetic clustered smoke data',
        split='Frozen before measurement. Default screen evaluates tune only; test remains unread by search.'))
    print('PREPARED', out, flush=True)


def load_workload(path, split):
    root = Path(path); manifest = json.loads((root/'manifest.json').read_text())
    if digest(root/'workload.npz') != manifest['workload_sha256']: raise ValueError('Workload hash mismatch')
    with np.load(root/'workload.npz', allow_pickle=False) as z: data = dict(z)
    indices = np.array(manifest[split+'_indices'], dtype='i8')
    for key in ['queries', 'query_ids', 'buckets']: data[key] = data[key][indices]
    return data, manifest


def aggregate(rows):
    result = []
    for condition, method in sorted({(r['condition'], r['method']) for r in rows}):
        rs = [r for r in rows if (r['condition'], r['method']) == (condition, method)]
        entry = dict(condition=condition, method=method, queries=len(rs))
        for key in ['item_recall', 'group_recall', 'fill', 'distance_evaluations', 'expanded']:
            values = [r[key] for r in rs if r.get(key) is not None]
            entry[key] = float(np.mean(values)) if values else None
        entry['p50_ms'] = float(np.median([r['ms'] for r in rs]))
        entry['p95_ms'] = float(np.quantile([r['ms'] for r in rs], .95))
        entry['violations'] = sum(sum(r[key] for key in ['invalid_ids', 'duplicate_ids', 'eligibility_violations', 'group_violations']) for r in rs)
        result.append(entry)
    return result


def export_upstream(a):
    data, manifest = load_workload(a.workload, a.split)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    np.savez(out/'workload.npz', **data)
    write_bin(out/'base.bin', data['vectors']); write_bin(out/'query.bin', data['queries'])
    np.savetxt(out/'groups.txt', data['groups'], fmt='%d')
    write_json(out/'config.json', dict(k=a.k, selectivities=a.selectivities, split=a.split))
    write_json(out/'manifest.json', dict(workload_sha256=digest(out/'workload.npz'),
        parent_workload_sha256=manifest['workload_sha256'], split=a.split))
    print('EXPORTED', out, flush=True)


def measure_graph(graph, q, mask, k, budgets, pool, cap, prefilter, restart):
    times = []; outputs = []; stats = []; total = 0.; evaluations = expanded = 0
    session = None
    try:
        for budget in budgets:
            start = time.perf_counter()
            if session is None: session = graph.session(q)
            session.advance(budget)
            got = session.select(mask, k, pool, cap, prefilter)
            current = session.stats()
            if restart:
                evaluations += current['distance_evaluations']; expanded += current['expanded']
                session.close(); session = None
            total += (time.perf_counter()-start)*1000
            outputs.append(got); times.append(total)
            stats.append(dict(current, distance_evaluations=evaluations if restart else current['distance_evaluations'],
                              expanded=expanded if restart else current['expanded']))
    finally:
        if session is not None: session.close()
    return outputs, times, stats


def screen(a):
    import faiss
    from threadpoolctl import threadpool_limits
    data, manifest = load_workload(a.workload, a.split)
    x, q, g = data['vectors'], data['queries'], data['groups']; n = len(x)
    if a.k > n or a.budgets[-1] > n: raise ValueError('k/budget exceeds database size')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', vars(a))
    rows = []; selections = []; rng = np.random.default_rng(20261116)
    with threadpool_limits(limits=1):
        w = Workload('D004', x, q, g, np.ones((len(q), n), bool), 'l2').validate()
        exact = NativeExact(x, g, a.cache_dir)
        h = HNSW(w, m=32, ef_construction=200, seed=a.index_seed)
        graph = Graph.from_hnsw(h.index, x, g, cache=a.cache_dir)
        write_json(out/'manifest.json', dict(schema=1, workload_sha256=manifest['workload_sha256'],
            numpy=np.__version__, faiss=faiss.__version__, python=platform.python_version(),
            source_sha256=digest(__file__), graph_native=graph.metadata, exact_native=exact.metadata,
            graph_hash=array_hash(graph.edges), hnsw_build_seconds=h.build_seconds,
            index_bytes=h.index_bytes, exported_edges_bytes=graph.edges.nbytes,
            split=a.split, query_ids=data['query_ids'].tolist(),
            timing='Single CPU thread; warmup + repetitions; median per query. Resume/restart rows are cumulative across checkpoints, including selection.',
            warning='Custom level-0 unbounded-frontier best-first control, fixed entry; not FAISS HNSW/ACORN/DiskANN search. Caps apply to candidate shortlist, NOT traversal.'))
        specs = [('exact',)] + [('faiss', ef) for ef in a.efs]
        specs += [('graph', pre, cap, restart) for pre in [False, True] for cap in a.group_caps for restart in [False, True]]
        for fraction in a.selectivities:
            condition = f'both-{fraction:g}'; masks = eligibility(data, fraction); w.allowed = masks
            print('START', condition, a.split, flush=True)
            for qi in range(len(q)):
                ref = exact.search(q[qi], masks[qi], a.k, True)
                samples = {}; final = {}
                for rep in range(a.repeats+1):
                    for si in rng.permutation(len(specs)):
                        spec = specs[si]
                        if spec[0] == 'graph':
                            _, pre, cap, restart = spec
                            got_list, times, counters = measure_graph(graph, q[qi], masks[qi], a.k,
                                a.budgets, a.pool, cap, pre, restart)
                            values = [(f'graph_{"restart" if restart else "resume"}_{"pre" if pre else "post"}_cap{cap}_b{b}', got, ms, st)
                                      for b, got, ms, st in zip(a.budgets, got_list, times, counters)]
                        else:
                            start = time.perf_counter()
                            if spec[0] == 'exact':
                                got = exact.search(q[qi], masks[qi], a.k, True); name = 'exact_cpp'
                            else:
                                got, _ = h.search(qi, a.k, spec[1], min(a.pool, n), min(a.hnsw_cap, n), adaptive=True, filtered=True)
                                name = f'faiss_native_restart_ef{spec[1]}'
                            values = [(name, got, (time.perf_counter()-start)*1000, {})]
                        for name, got, ms, st in values:
                            if rep: samples.setdefault(name, []).append(ms)
                            final[name] = (got, st)
                for name, (got, st) in final.items():
                    rows.append(dict(condition=condition, method=name, query=qi, source_query=int(data['query_ids'][qi]),
                        split=a.split, ms=float(np.median(samples[name])), **st,
                        **evaluate(x, q[qi], g, masks[qi], got, ref, True)))
                    selections.append(dict(condition=condition, method=name, query=qi, ids=got.tolist(), reference=ref.tolist()))
            write_json(out/f'checkpoint-{condition}.json', [r for r in rows if r['condition']==condition])
        write_json(out/'per_query.json', rows); write_json(out/'selections.json', selections)
        write_json(out/'summary.json', aggregate(rows)); print('COMPLETE', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', choices=['sift', 'clustered'], default='clustered')
    prep.add_argument('--n', type=int, default=1000); prep.add_argument('--old-queries', type=int, default=64)
    prep.add_argument('--queries', type=int, default=16); prep.add_argument('--data-seed', type=int, default=20261115)
    prep.add_argument('--query-seed', type=int, default=20261116); prep.add_argument('--data-dir', default='external/sift1m')
    prep.add_argument('--out', required=True)
    export = sub.add_parser('export-upstream')
    export.add_argument('--workload', required=True); export.add_argument('--out', required=True)
    export.add_argument('--split', choices=['tune', 'test'], default='tune')
    export.add_argument('--k', type=int, default=100)
    export.add_argument('--selectivities', nargs='+', type=float, default=[.1,.01])
    run = sub.add_parser('screen'); run.add_argument('--workload', required=True)
    run.add_argument('--out', required=True); run.add_argument('--split', choices=['tune', 'test'], default='tune')
    run.add_argument('--k', type=int, default=10); run.add_argument('--budgets', nargs='+', type=int, default=[32,128,512])
    run.add_argument('--pool', type=int, default=200); run.add_argument('--group-caps', nargs='+', type=int, default=[1,4,16,0])
    run.add_argument('--efs', nargs='+', type=int, default=[128,512]); run.add_argument('--hnsw-cap', type=int, default=5000)
    run.add_argument('--selectivities', nargs='+', type=float, default=[.1,.01]); run.add_argument('--repeats', type=int, default=3)
    run.add_argument('--index-seed', type=int, default=11); run.add_argument('--cache-dir', default='.local/d004-native')
    a = p.parse_args()
    if a.command == 'prepare':
        if a.n < 100 or not 2 <= a.queries <= 10000 or a.old_queries < 1 or min(a.data_seed,a.query_seed)<0:
            p.error('Invalid dimensions/seeds')
        prepare(a)
    elif a.command == 'export-upstream':
        if a.k<1 or not all(0<f<1 for f in a.selectivities): p.error('Invalid k/selectivities')
        export_upstream(a)
    else:
        if (a.k < 1 or a.pool < a.k or a.hnsw_cap < a.pool or a.repeats < 1 or min(a.budgets) < 1
                or a.budgets != sorted(set(a.budgets)) or min(a.group_caps)<0 or min(a.efs)<1
                or not all(0<f<=1 for f in a.selectivities)):
            p.error('Invalid budgets/caps/selectivities')
        screen(a)


if __name__ == '__main__': main()
