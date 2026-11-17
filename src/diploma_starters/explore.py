"""D-006: frozen natural workload and small, explicitly diagnostic policy screen."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import time
import zipfile

import numpy as np

from .assets import fetch
from .ann_data import read_ivecs
from .group_screen import NativeExact, digest, fvecs, write_bin, evaluate

ROOT = Path(__file__).resolve().parents[2]
REV = '659041e1b12e3ab41c70ad12d11a29a883f17e15'
COUNTERS = ('distances', 'expansions', 'eligible_discovered', 'groups_discovered',
            'representative_improvements', 'ineligible_discovered',
            'repeat_group_expansions', 'peak_pending')
# One mechanism at a time; reserve has a matching no-cap control.
POLICIES = {
    'ordinary': (0, 0, 'single'),
    'cap1': (1, 0, 'single'),
    'cap4': (4, 0, 'single'),
    'reserve': (0, 32, 'single'),
    'cap1_reserve': (1, 32, 'single'),
    'random4': (0, 0, 'random4'),
    'balanced4': (0, 0, 'balanced4'),
}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_receipt():
    return {str(p.relative_to(ROOT)): digest(p) for p in [
        Path(__file__), ROOT/'native/explore_groups.cpp',
        ROOT/'native/exact_groups.cpp', ROOT/'src/diploma_starters/group_screen.py',
        ROOT/'src/diploma_starters/ann_data.py', ROOT/'src/diploma_starters/assets.py',
        ROOT/'evidence/sources.lock.json']}


def metadata(rows):
    """Category *sets* form a partition; dates deliberately need no epoch assumption."""
    keys = [tuple(sorted(set(r['sub_categories']))) for r in rows]
    if any(not key for key in keys):
        raise ValueError('Empty category set is not a supported natural group')
    lookup = {key: i for i, key in enumerate(sorted(set(keys)))}
    groups = np.array([lookup[key] for key in keys], dtype='i8')
    counts = np.array([r['number_of_sub_categories'] for r in rows], dtype='i8')
    dates = np.array([r['update_date'] for r in rows], dtype='i8')
    # The source count is used for published EM replay; do not silently rewrite it.
    return groups, counts, dates


def split_ids(n, seed, count):
    if count < 1 or 2*count > n:
        raise ValueError('Need disjoint development and held-out queries')
    ids = np.random.default_rng(seed).permutation(n)
    return ids[:count], ids[count:2*count]


def read_diskann(path, n):
    """Pinned upstream src/in_mem_graph_store.cpp, 64-bit little-endian format."""
    path = Path(path)
    with path.open('rb') as f:
        header = f.read(24)
        if len(header) != 24:
            raise ValueError('Truncated DiskANN header')
        size, degree, entry, frozen = struct.unpack('<QIIQ', header)
        if size != path.stat().st_size or frozen or entry >= n or degree > n:
            raise ValueError('Unsupported or corrupt DiskANN graph header')
        edges = np.full((n, max(1, degree)), -1, dtype='i8')
        for i in range(n):
            b = f.read(4)
            if len(b) != 4:
                raise ValueError('Truncated DiskANN graph')
            width, = struct.unpack('<I', b)
            raw = f.read(4*width)
            if width > degree or len(raw) != 4*width:
                raise ValueError('Invalid DiskANN degree')
            ids = np.frombuffer(raw, dtype='<u4')
            if np.any(ids >= n):
                raise ValueError('Invalid DiskANN neighbor')
            edges[i, :width] = ids
        if f.read(1):
            raise ValueError('Unexpected graph rows/trailing bytes')
    return edges, int(entry)


class Kernel:
    def __init__(self, x, edges, cache):
        self.x = np.ascontiguousarray(x, dtype='f4')
        self.edges = np.ascontiguousarray(edges, dtype='i8')
        if (self.x.ndim != 2 or not all(self.x.shape) or not np.isfinite(self.x).all()
                or self.edges.ndim != 2 or self.edges.shape[0] != len(x)
                or not self.edges.shape[1] or np.any(self.edges < -1)
                or np.any(self.edges >= len(x))):
            raise ValueError('Invalid graph/vectors')
        source = ROOT/'native/explore_groups.cpp'
        compiler = os.environ.get('CXX', 'g++')
        version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
        key = identity([digest(source), version, platform.machine()])[:20]
        cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
        binary = cache/f'explore-{key}.so'
        if not binary.exists():
            temp = cache/f'explore-{key}-{os.getpid()}.so'
            try:
                subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                                str(source), '-o', str(temp)], check=True)
                temp.replace(binary)
            finally:
                temp.unlink(missing_ok=True)
        self.lib = ctypes.CDLL(str(binary))
        ptr = lambda t: np.ctypeslib.ndpointer(t, flags='C_CONTIGUOUS')
        self.fn = self.lib.explore_groups
        self.fn.argtypes = [ptr(np.float32), ptr(np.float32), ptr(np.int64),
            ptr(np.int64), ptr(np.uint8), ctypes.c_int64, ctypes.c_int, ctypes.c_int,
            ptr(np.int64)] + [ctypes.c_int]*6 + [ptr(np.int64), ptr(np.int64)]
        self.fn.restype = ctypes.c_int
        self.receipt = dict(compiler=version, source_sha256=digest(source),
                            binary_sha256=digest(binary), accumulation='float64')

    def search(self, q, groups, mask, starts, budget, beam, cap, reserve, k):
        q = np.ascontiguousarray(q, dtype='f4')
        groups = np.ascontiguousarray(groups, dtype='i8')
        mask = np.ascontiguousarray(mask, dtype='u1')
        starts = np.ascontiguousarray(starts, dtype='i8')
        n, d = self.x.shape
        if (q.shape != (d,) or not np.isfinite(q).all() or groups.shape != (n,)
            or np.any(groups < 0) or np.any(groups >= n) or mask.shape != (n,)
            or starts.ndim != 1 or not len(starts) or np.any(starts < 0)
            or np.any(starts >= n) or len(np.unique(starts)) != len(starts)
            or budget < len(starts) or beam <= reserve or reserve < 0 or cap < 0 or k < 1):
            raise ValueError('Invalid search arguments')
        out, stats = np.empty(k, dtype='i8'), np.zeros(8, dtype='i8')
        size = self.fn(self.x, q, self.edges, groups, mask, n, d, self.edges.shape[1],
                       starts, len(starts), budget, beam, cap, reserve, k, out, stats)
        if size < 0:
            raise RuntimeError(f'Native search failed: {size}')
        return out[:size], dict(zip(COUNTERS, map(int, stats)))


def select_candidates(x, q, groups, mask, ids, k):
    ids = np.unique(np.asarray(ids, dtype='i8'))
    ids = ids[(ids >= 0) & (ids < len(x))]
    ids = ids[mask[ids]]
    # Direct float64 L2 matches the reference accumulation closely; only shortlist copied.
    scores = np.square(x[ids].astype('f8')-q.astype('f8')).sum(axis=1)
    result, seen = [], set()
    for pos in np.lexsort((ids, scores)):
        node = int(ids[pos]); group = int(groups[node])
        if group not in seen:
            result.append(node); seen.add(group)
            if len(result) == k:
                break
    return np.array(result, dtype='i8')


def graph_audit(edges, groups):
    degree = np.sum(edges >= 0, axis=1)
    unique = np.array([len(np.unique(groups[row[row >= 0]])) for row in edges])
    return dict(mean_degree=float(degree.mean()), max_degree=int(degree.max()),
                mean_neighbor_groups=float(unique.mean()), bytes=int(edges.nbytes))


def checkpoint(out, name, files, info=None):
    path = out/f'{name}-receipt.json'
    if info is not None:
        atomic(path, dict(info=info, files={p.name:digest(p) for p in files}))
        return info
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    for filename, expected in receipt['files'].items():
        if not (out/filename).exists() or digest(out/filename) != expected:
            raise ValueError('Build checkpoint changed: '+filename)
    return receipt['info']


def prepare(a):
    import faiss
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    config = dict(dataset=a.dataset, queries=a.queries, seed=a.seed, faiss=faiss.__version__,
                  build_threads=a.build_threads, diskann=str(Path(a.diskann).resolve()) if a.diskann else None)
    cfg = out/'prepare.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Preparation configuration differs; use a new output directory')
    atomic(cfg, config)
    if (out/'manifest.json').exists():
        print('Prepared workload already exists; run it or choose another directory.', flush=True)
        return
    if a.diskann:
        upstream = Path(a.diskann).resolve()
        revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', '-C', str(upstream), 'diff', 'HEAD', '--'], text=True)
        executable = upstream/'build/apps/build_memory_index'
        if revision != REV or changed or not executable.is_file():
            raise ValueError('Require clean pinned Diverse DiskANN checkout and built build_memory_index')
    lock = json.loads((ROOT/'evidence/sources.lock.json').read_text())['assets']
    data_dir = Path(a.data_dir)/a.dataset
    receipts = {}
    for suffix in ('database_vectors', 'query_vectors', 'database_attributes',
                   'em_query_attributes', 'ground_truth_em'):
        spec = lock[f'fanns_{a.dataset}_{suffix}']
        print('FETCH/VERIFY', spec['filename'], flush=True)
        fetch(spec, data_dir/spec['filename']); receipts[spec['filename']] = spec
    x = np.ascontiguousarray(fvecs(data_dir/'database_vectors.fvecs'))
    all_q = fvecs(data_dir/'query_vectors.fvecs')
    rows = [json.loads(s) for s in (data_dir/'database_attributes.jsonl').read_text().splitlines()]
    labels = np.array([json.loads(s)['label'] for s in
                       (data_dir/'em_query_attributes.jsonl').read_text().splitlines()], dtype='i8')
    if len(rows) != len(x) or len(labels) != len(all_q):
        raise ValueError('Metadata/vector row mismatch')
    groups, counts, dates = metadata(rows)
    dev, holdout = split_ids(len(all_q), a.seed, a.queries)
    np.save(out/'vectors.npy', x)
    np.savez(out/'workload.npz', queries=np.ascontiguousarray(all_q[dev]), groups=groups,
             counts=counts, dates=dates, labels=labels[dev], query_ids=dev,
             holdout_query_ids=holdout)
    supplied = read_ivecs(data_dir/'ground_truth_em.ivecs')
    atomic(out/'published_ground_truth.json', [supplied[int(i)].tolist() for i in dev])
    rng = np.random.default_rng(a.seed + 1)
    random_starts = rng.choice(len(x), min(4, len(x)), replace=False).tolist()
    chosen_groups = rng.choice(np.unique(groups), min(4, len(np.unique(groups))), replace=False)
    balanced_starts = [int(rng.choice(np.flatnonzero(groups == g))) for g in chosen_groups]
    manifest = dict(schema=1, config=config, sources=receipts, code=source_receipt(),
                    shape=list(x.shape), faiss=faiss.__version__, numpy=np.__version__,
                    python=platform.python_version(), machine=platform.machine(),
                    query_ids=dev.tolist(), holdout_query_ids=holdout.tolist(),
                    entry_sets=dict(random4=random_starts, balanced4=balanced_starts), graphs={})
    faiss.omp_set_num_threads(a.build_threads)
    info = checkpoint(out,'hnsw0',[])
    if info is None:
        print('BUILD HNSW M=32 efConstruction=200', flush=True)
        index = faiss.IndexHNSWFlat(x.shape[1], 32, faiss.METRIC_L2)
        index.hnsw.efConstruction = 200; index.hnsw.rng = faiss.RandomGenerator(a.seed)
        t = time.perf_counter(); index.add(x); seconds = time.perf_counter()-t
        faiss.write_index(index, str(out/'hnsw.faiss'))
        offsets = faiss.vector_to_array(index.hnsw.offsets)
        neighbors = faiss.vector_to_array(index.hnsw.neighbors)
        width = index.hnsw.nb_neighbors(0)
        edges = np.array([neighbors[int(o):int(o)+width] for o in offsets[:-1]], dtype='i8')
        np.save(out/'hnsw0.npy', edges)
        info = checkpoint(out,'hnsw0',[out/'hnsw.faiss',out/'hnsw0.npy'],
            dict(entry=int(index.hnsw.entry_point), build_seconds=seconds,
                 graph_sha256=digest(out/'hnsw0.npy'), **graph_audit(edges, groups)))
        print('BUILT HNSW seconds=', round(seconds, 2), flush=True)
        del index
    else:
        print('REUSE verified HNSW build',flush=True)
    manifest['graphs']['hnsw0'] = info
    if a.diskann:
        write_bin(out/'base.bin', x); np.savetxt(out/'groups.txt', groups, fmt='%d')
        for name, diverse in [('diskann', False), ('diverse_diskann', True)]:
            info = checkpoint(out,name,[])
            if info is not None:
                manifest['graphs'][name] = info
                print('REUSE verified',name,'build',flush=True)
                continue
            prefix = out/name
            command = [str(executable), '--data_type', 'float', '--dist_fn', 'l2',
                '--data_path', str(out/'base.bin'), '--index_path_prefix', str(prefix),
                '-R', '64', '-L', '200', '--alpha', '1.2', '-T', str(a.build_threads)]
            if diverse:
                command += ['--seller_file', str(out/'groups.txt'), '--NumDiverse', '16']
            print('BUILD', name, flush=True); t = time.perf_counter()
            with (out/f'{name}-build.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            seconds = time.perf_counter()-t
            edges, entry = read_diskann(prefix, len(x)); np.save(out/f'{name}.npy', edges)
            manifest['graphs'][name] = checkpoint(out,name,[prefix,out/f'{name}.npy'],
                dict(entry=entry, command=command, revision=revision,
                build_seconds=seconds, graph_sha256=digest(out/f'{name}.npy'), **graph_audit(edges, groups)))
            print('BUILT', name, 'seconds=', round(seconds, 2), flush=True)
    manifest['files'] = {p.name: digest(p) for p in [out/'vectors.npy', out/'workload.npz',
                         out/'published_ground_truth.json', out/'hnsw.faiss']}
    manifest['date_thresholds'] = {str(f): int(np.quantile(dates, f, method='lower')) for f in (.1, .5)}
    atomic(out/'manifest.json', manifest)
    print('PREPARED', out, flush=True)


def conditions(data, qi, thresholds):
    n = len(data['groups'])
    yield 'group_all', np.ones(n, dtype=bool), True
    em = data['counts'] == data['labels'][qi]
    yield 'em_filter', em, False
    yield 'em_group', em, True
    for fraction, threshold in thresholds.items():
        mask = data['dates'] <= threshold
        yield f'date{fraction}_filter', mask, False
        yield f'date{fraction}_group', mask, True


def replay(x, q, mask, supplied, exact, k):
    """Report ties separately; any non-tie disagreement stops the experiment."""
    expected = np.asarray(supplied[:k], dtype='i8')
    got = exact.search(q, mask, k, False)
    if (len(expected) != len(got) or np.any(expected < 0) or np.any(expected >= len(x))
        or len(np.unique(expected)) != len(expected) or not mask[expected].all()):
        raise ValueError('Invalid supplied EM ground truth')
    dist = lambda ids: np.sort(np.square(x[ids].astype('f8')-q.astype('f8')).sum(axis=1))
    same = set(expected.tolist()) == set(got.tolist())
    close = np.allclose(dist(expected), dist(got), rtol=1e-6, atol=1e-9)
    if not same and not close:
        raise ValueError('Published EM replay differs beyond numeric/tie tolerance')
    return dict(id_set_equal=same, distances_within_tolerance=bool(close), k=k,
                expected=expected.tolist(), actual=got.tolist())


def summarize(rows):
    result = []
    keys = sorted({(r['condition'], r['k'], r['method'], r['budget']) for r in rows})
    for key in keys:
        selected = [r for r in rows if (r['condition'],r['k'],r['method'],r['budget']) == key]
        item = dict(zip(('condition','k','method','budget'), key)); item['queries'] = len(selected)
        for field in ('item_recall','group_recall','fill','distance_ratio','mean_l2_regret','ms'):
            values = [r[field] for r in selected if r[field] is not None]
            item[field] = float(np.mean(values)) if values else None
        item['feasible_queries'] = sum(r['feasible'] for r in selected)
        item['violations'] = sum(sum(r[f] for f in ('eligibility_violations','group_violations',
                                    'invalid_ids','duplicate_ids')) for r in selected)
        result.append(item)
    return result


def frontier(table):
    """Observed grid frontier, not interpolation or held-out model selection."""
    result = []
    for condition,k,method in sorted({(r['condition'],r['k'],r['method']) for r in table}):
        candidates = [r for r in table if (r['condition'],r['k'],r['method']) == (condition,k,method)
            and r['feasible_queries'] == r['queries'] and r['violations'] == 0
            and r['item_recall'] is not None and r['item_recall'] >= .95
            and r['group_recall'] >= .95 and r['fill'] >= .99]
        result.append(dict(condition=condition,k=k,method=method,
            best_at_mean_recall_95=min(candidates,key=lambda r:r['ms']) if candidates else None))
    return result


def run(a):
    import faiss
    from threadpoolctl import threadpool_limits
    out = Path(a.out); manifest = json.loads((out/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(out/name) != expected:
            raise ValueError('Prepared file changed: '+name)
    for name, info in manifest['graphs'].items():
        if digest(out/f'{name}.npy') != info['graph_sha256']:
            raise ValueError('Graph changed: '+name)
    if faiss.__version__ != manifest['faiss']:
        raise ValueError('FAISS version differs from preparation')
    x = np.load(out/'vectors.npy', mmap_mode='r')
    data = dict(np.load(out/'workload.npz'))
    nq = a.query_count or len(data['queries'])
    if not 1 <= nq <= len(data['queries']):
        raise ValueError('Query count exceeds frozen development split')
    directory = out/'runs'/a.name; directory.mkdir(parents=True, exist_ok=True)
    plan = dict(manifest=digest(out/'manifest.json'), code=source_receipt(),
        query_count=nq, budgets=a.budgets, ks=a.ks, beam=a.beam, repeats=a.repeats,
        policies=a.policies, phase=a.phase, threads=1)
    plan_path = directory/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Run plan changed; use a new --name')
    atomic(plan_path, plan)
    # Fail concurrent use, release on normal exit/exception. After SIGKILL remove lock manually.
    lock = directory/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        with threadpool_limits(limits=1):
            faiss.omp_set_num_threads(1)
            _run_cells(a, out, directory, manifest, data, x, nq)
    finally:
        lock.unlink(missing_ok=True)


def _run_cells(a, out, directory, manifest, data, x, nq):
    import faiss
    index = faiss.read_index(str(out/'hnsw.faiss'))
    exact = NativeExact(x, data['groups'], a.cache_dir)
    kernels = {name: Kernel(x, np.load(out/f'{name}.npy'), a.cache_dir)
               for name in manifest['graphs']}
    atomic(directory/'environment.json', dict(python=platform.python_version(),
        platform=platform.platform(), numpy=np.__version__, faiss=faiss.__version__,
        exact=exact.metadata, kernels={name:k.receipt for name,k in kernels.items()}))
    supplied = json.loads((out/'published_ground_truth.json').read_text())
    rows = []; started = time.perf_counter()
    all_groups = data['groups']; individual = np.arange(len(x), dtype='i8')
    replay_rows = []
    for qi in range(nq):
        q = data['queries'][qi]
        em = data['counts'] == data['labels'][qi]
        replay_rows.append(replay(x,q,em,supplied[qi],exact,min(100,len(supplied[qi]))))
        atomic(directory/'em_replay.json', replay_rows)
        for condition, mask, grouped in conditions(data, qi, manifest['date_thresholds']):
            groups = all_groups if grouped else individual
            eligible_groups = len(np.unique(groups[mask]))
            total_per_group = np.bincount(all_groups, minlength=len(x))
            eligible_per_group = np.bincount(all_groups[mask], minlength=len(x))
            split_groups = int(np.sum((eligible_per_group > 0) & (eligible_per_group < total_per_group)))
            for k in a.ks:
                cell = directory/'cells'/f'q{qi:04d}-{condition}-k{k}.json'
                if cell.exists():
                    saved = json.loads(cell.read_text())
                    if saved['plan'] != identity(json.loads((directory/'plan.json').read_text())):
                        raise ValueError('Cell plan receipt mismatch')
                    if saved['checksum'] != identity(saved['rows']):
                        raise ValueError('Cell checksum mismatch')
                    rows.extend(saved['rows']); continue
                reference = exact.search(q,mask,k,grouped)
                methods = []
                if a.phase in ('baseline','all'):
                    methods += [('exact',0,None)]
                    for budget in a.budgets:
                        methods += [('hnsw_post',budget,None),('hnsw_filter',budget,None)]
                if a.phase in ('explore','all') and grouped:
                    for graph in kernels:
                        for policy in a.policies:
                            for budget in a.budgets:
                                methods.append((graph+'/'+policy,budget,(graph,policy)))
                if not methods:
                    continue
                results, timings = {}, {m[:2]: [] for m in methods}
                order = np.random.default_rng(10000+qi*101+k)
                for repeat in range(a.repeats+1):
                    for pos in order.permutation(len(methods)):
                        method,budget,diagnostic = methods[pos]
                        t = time.perf_counter(); stats = {}
                        if method == 'exact':
                            got = exact.search(q,mask,k,grouped)
                        elif method.startswith('hnsw_'):
                            pool = min(len(x), max(k,budget))
                            params = faiss.SearchParametersHNSW(); params.efSearch = pool
                            if method == 'hnsw_filter':
                                bitmap = np.packbits(mask,bitorder='little')
                                params.sel = faiss.IDSelectorBitmap(len(x),faiss.swig_ptr(bitmap))
                            _, ids = index.search(q[None,:],pool,params=params)
                            got = select_candidates(x,q,groups,mask,ids[0],k)
                            stats = dict(pool=pool, effective_ef=pool,
                                         returned_candidates=int(np.sum(ids[0]>=0)))
                        else:
                            graph, policy = diagnostic
                            cap,reserve,entry = POLICIES[policy]
                            starts = ([manifest['graphs'][graph]['entry']] if entry == 'single'
                                      else manifest['entry_sets'][entry])
                            got,stats = kernels[graph].search(q,groups,mask,starts,budget,a.beam,cap,reserve,k)
                        elapsed = (time.perf_counter()-t)*1000
                        key = method,budget
                        if repeat:
                            timings[key].append(elapsed)
                        if key in results and not np.array_equal(results[key][0],got):
                            raise ValueError('Nondeterministic repeated search')
                        results[key] = got,stats
                cell_rows = []
                for method,budget,_ in methods:
                    got,stats = results[(method,budget)]
                    metrics = evaluate(x,q,groups,mask,got,reference,grouped)
                    if any(metrics[f] for f in ('eligibility_violations','group_violations','invalid_ids','duplicate_ids')):
                        raise ValueError('Retrieval constraint violation')
                    row = dict(condition=condition,k=k,method=method,budget=budget,
                        query_id=int(data['query_ids'][qi]), eligible_items=int(mask.sum()),
                        eligible_groups=eligible_groups, groups_cut_by_filter=split_groups,
                        feasible=eligible_groups>=k, ms=float(np.median(timings[(method,budget)])),
                        samples_ms=timings[(method,budget)], counters=stats,
                        selected=got.tolist(),reference=reference.tolist(),**metrics)
                    cell_rows.append(row)
                atomic(cell,dict(plan=identity(json.loads((directory/'plan.json').read_text())),
                                rows=cell_rows,checksum=identity(cell_rows)))
                rows.extend(cell_rows)
            print(f'COMPLETE query {qi+1}/{nq} {condition}',flush=True)
        atomic(directory/'summary.json',dict(status='running',completed_queries=qi+1,
                    wall_seconds_this_invocation=time.perf_counter()-started,table=summarize(rows)))
    summary = dict(status='complete', completed_queries=nq,
                  wall_seconds_this_invocation=time.perf_counter()-started,table=summarize(rows))
    summary['frontier'] = frontier(summary['table'])
    atomic(directory/'summary.json',summary)
    lines = ['# D-006 development screen', '',
        'Diagnostic level-zero traversal is not a published HNSW/ACORN/RACORN implementation.',
        'Budget means distance evaluations for diagnostic kernels; shortlist/ef for FAISS.',
        'Latency includes mask packing, traversal, result selection and per-query allocations.',
        'Masks are precomputed; predicate evaluation/index lookup cost is excluded for all methods.',
        'No held-out queries were evaluated. No p95/p99 claim from this small screen.', '',
        '| Condition | k | Method | Budget | Item recall | Group recall | Mean ms | Feasible |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summary['table']:
        fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
        lines.append(f"| {r['condition']} | {r['k']} | {r['method']} | {r['budget']} | "
                     f"{fmt(r['item_recall'])} | {fmt(r['group_recall'])} | {r['ms']:.3f} | "
                     f"{r['feasible_queries']}/{r['queries']} |")
    (directory/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',directory,flush=True)


def package(a):
    out = Path(a.out); run = out/'runs'/a.name
    destination = Path(a.destination)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        files = [out/'manifest.json',out/'prepare.json']
        files += sorted(out.glob('*-build.log'))
        files += sorted(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.md'))
        for p in files:
            z.write(p,str(p.relative_to(out)))
    print(destination,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--out',required=True)
    prep.add_argument('--dataset',choices=['small','medium'],default='medium')
    prep.add_argument('--data-dir',default='.local/d006-data')
    prep.add_argument('--queries',type=int,default=32)
    prep.add_argument('--seed',type=int,default=29017)
    prep.add_argument('--build-threads',type=int,default=1)
    prep.add_argument('--diskann',help='Optional clean pinned checkout with built executables')
    runp = sub.add_parser('run')
    runp.add_argument('--out',required=True); runp.add_argument('--name',required=True)
    runp.add_argument('--query-count',type=int,default=0)
    runp.add_argument('--phase',choices=['baseline','explore','all'],default='all')
    runp.add_argument('--budgets',type=int,nargs='+',default=[256,1024,4096])
    runp.add_argument('--ks',type=int,nargs='+',default=[10,50,100])
    runp.add_argument('--beam',type=int,default=128)
    runp.add_argument('--repeats',type=int,default=3)
    runp.add_argument('--policies',nargs='+',choices=list(POLICIES),default=list(POLICIES))
    runp.add_argument('--cache-dir',default='.local/d006-native')
    pack = sub.add_parser('package'); pack.add_argument('--out',required=True)
    pack.add_argument('--name',required=True); pack.add_argument('--destination',required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        if a.queries<1 or a.build_threads<1:p.error('Positive query/thread counts required')
        prepare(a)
    elif a.command == 'run':
        if (not a.name or Path(a.name).name != a.name or a.name in ('.','..') or
            min(a.budgets)<4 or min(a.ks)<1 or a.repeats<1 or a.beam<=32 or
            len(set(a.budgets))!=len(a.budgets) or len(set(a.ks))!=len(a.ks) or
            len(set(a.policies))!=len(a.policies)):
            p.error('Invalid run name or budget/k/beam/repeat/policy configuration')
        run(a)
    else:
        if Path(a.name).name != a.name or a.name in ('.','..'):p.error('Invalid run name')
        package(a)


if __name__ == '__main__':
    main()
