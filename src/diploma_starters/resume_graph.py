"""Persistent best-first level-0 traversal: a control, not an HNSW reproduction."""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
from .group_screen import ROOT, digest


class Graph:
    def __init__(self, vectors, edges, groups, entry=0, cache='.local/d004-native'):
        self.x = np.ascontiguousarray(vectors, dtype='f4')
        self.edges = np.ascontiguousarray(edges, dtype='i8')
        self.groups = np.ascontiguousarray(groups, dtype='i8')
        n = len(self.x)
        if (self.x.ndim != 2 or not all(self.x.shape) or not np.isfinite(self.x).all()
                or self.edges.ndim != 2 or self.edges.shape[0] != n or not self.edges.shape[1]
                or self.groups.shape != (n,) or np.any(self.groups < 0) or np.any(self.groups >= n)
                or np.any(self.edges < -1) or np.any(self.edges >= n) or not 0 <= entry < n):
            raise ValueError('Invalid graph arrays or entry')
        self.entry = int(entry)
        compiler = os.environ.get('CXX', 'g++')
        source = ROOT / 'native/resume_graph.cpp'
        version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
        key = hashlib.sha256((digest(source)+version+platform.machine()).encode()).hexdigest()[:20]
        path = Path(cache) / f'resume-{key}.so'; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(f'.{os.getpid()}.so')
            try:
                subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(tmp)], check=True)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        self.lib = ctypes.CDLL(str(path)); ptr = np.ctypeslib.ndpointer
        f32 = ptr(np.float32, flags='C_CONTIGUOUS'); i64 = ptr(np.int64, flags='C_CONTIGUOUS')
        u8 = ptr(np.uint8, flags='C_CONTIGUOUS'); handle = ctypes.c_void_p
        self.lib.rg_create.argtypes = [f32, f32, i64, ctypes.c_int64, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
        self.lib.rg_create.restype = handle
        self.lib.rg_free.argtypes = [handle]; self.lib.rg_free.restype = None
        self.lib.rg_advance.argtypes = [handle, ctypes.c_int64]; self.lib.rg_advance.restype = ctypes.c_int
        self.lib.rg_select.argtypes = [handle, i64, u8, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, i64]
        self.lib.rg_select.restype = ctypes.c_int
        self.lib.rg_stats.argtypes = [handle, i64]; self.lib.rg_stats.restype = None
        self.metadata = dict(source_sha256=digest(source), binary_sha256=digest(path), compiler=version,
                             flags=['-O3', '-std=c++17', '-shared', '-fPIC'], accumulation='float64')

    @classmethod
    def from_hnsw(cls, index, vectors, groups, **kwargs):
        import faiss
        h = index.hnsw
        offsets = faiss.vector_to_array(h.offsets).astype('i8')
        all_edges = faiss.vector_to_array(h.neighbors)
        width = h.nb_neighbors(0)
        edges = np.array([all_edges[o:o+width] for o in offsets[:-1]], dtype='i8')
        return cls(vectors, edges, groups, h.entry_point, **kwargs)

    def session(self, query):
        return Session(self, query)


class Session:
    def __init__(self, graph, query):
        self.graph = graph
        q = np.ascontiguousarray(query, dtype='f4')
        if q.shape != (graph.x.shape[1],) or not np.isfinite(q).all():
            raise ValueError('Invalid query')
        self.handle = graph.lib.rg_create(graph.x, q, graph.edges, len(graph.x), graph.x.shape[1], graph.edges.shape[1], graph.entry)
        if not self.handle:
            raise MemoryError('Cannot allocate graph traversal')

    def close(self):
        if getattr(self, 'handle', None):
            self.graph.lib.rg_free(self.handle); self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def __del__(self): self.close()

    def advance(self, budget):
        if not self.handle or budget < 0: raise ValueError('Closed session or negative budget')
        if self.graph.lib.rg_advance(self.handle, budget): raise ValueError('Advance failed or decreasing budget')

    def select(self, mask, k, pool, cap=0, prefilter=True):
        mask = np.ascontiguousarray(mask, dtype='u1')
        if not self.handle or mask.shape != (len(self.graph.x),) or k < 1 or pool < 1 or cap < 0:
            raise ValueError('Invalid selection arguments')
        out = np.empty(k, dtype='i8')
        count = self.graph.lib.rg_select(self.handle, self.graph.groups, mask, k, pool, cap, int(prefilter), out)
        if count < 0: raise RuntimeError('Native selection failed')
        return out[:count]

    def stats(self):
        if not self.handle: raise ValueError('Closed session')
        out = np.empty(4, dtype='i8'); self.graph.lib.rg_stats(self.handle, out)
        return dict(zip(['expanded', 'distance_evaluations', 'frontier', 'peak_frontier'], map(int, out)))
