import importlib.util
import itertools
import tempfile
import unittest
from pathlib import Path

AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ['numpy','faiss','threadpoolctl'])
if AVAILABLE:
    import numpy as np
    from diploma_starters.ann import Workload, HNSW, exact, select, evaluate_selection, benchmark, collapse_identical_groups
    from diploma_starters.ann_data import read_fvecs, read_ivecs


@unittest.skipUnless(AVAILABLE,'Install .[ann] for Topic D tests')
class ANN(unittest.TestCase):
    def workload(self):
        return Workload('tiny',np.array([[2,0],[1,0],[0,1],[-1,0]],dtype='float32'),
                        np.array([[1,0],[0,1]],dtype='float32'),np.array([0,0,2,3]),
                        np.array([[True]*4,[False,True,True,True]])).validate()

    def test_partition_selection_matches_exhaustive_objective(self):
        rng=np.random.default_rng(9)
        for _ in range(20):
            values=rng.normal(size=8).astype('float32');groups=rng.integers(0,4,size=8)
            allowed=rng.random(8)>.3;k=3
            got=select(np.arange(8),values,groups,allowed,k)
            feasible=[c for size in range(k+1) for c in itertools.combinations(range(8),size)
                      if all(allowed[list(c)]) and len(set(groups[list(c)]))==len(c)]
            expected=max(feasible,key=lambda c:(len(c),sum(values[list(c)])))
            self.assertEqual(len(got),len(expected))
            self.assertAlmostEqual(float(values[got].sum()),float(values[list(expected)].sum()),places=5)

    def test_empty_mask_not_fake_perfect_recall(self):
        w=self.workload();w.allowed[0]=False
        self.assertEqual(len(exact(w,0,3)),0)
        self.assertIsNone(evaluate_selection(w,0,np.array([],dtype=int),np.array([],dtype=int))['group_recall'])

    def test_ties_are_stable_and_groups_not_counted_twice(self):
        got=select([3,1,2,0],[1,1,1,1],np.array([0,0,2,3]),np.ones(4,bool),2)
        self.assertEqual(got.tolist(),[0,2])

    def test_missing_candidate_sentinel_ignored(self):
        w=self.workload()
        self.assertEqual(select([-1,2],[999,1],w.groups,w.allowed[0],2).tolist(),[2])

    def test_ip_not_silently_normalized(self):
        w=self.workload();self.assertEqual(exact(w,0,1).tolist(),[0])
        self.assertEqual(float(w.vectors[0,0]),2.0)

    def test_known_l2_order(self):
        w=self.workload();w.metric='l2'
        self.assertEqual(exact(w,0,1).tolist(),[1])

    def test_native_filter_and_group_cap(self):
        w=self.workload();h=HNSW(w,m=4,ef_construction=20)
        for qi in [0,1]:
            got,_=h.search(qi,3,32,4,4,filtered=True)
            self.assertEqual(set(w.groups[got]),set(w.groups[exact(w,qi,3)]))
            self.assertTrue(w.allowed[qi,got].all())

    def test_gold_never_changes_index_or_retrieval(self):
        w=self.workload();h=HNSW(w,m=4,ef_construction=20)
        first,_=h.search(0,2,32,4,4)
        w.gold=[{'evidence':[3]},{'evidence':[0]}]
        second,_=h.search(0,2,32,4,4)
        self.assertEqual(first.tolist(),second.tolist())

    def test_filter_and_exact_group_member_representative(self):
        w=self.workload();w.allowed[0,0]=False
        self.assertEqual(exact(w,0,2).tolist(),[1,2])

    def test_adaptive_budget_is_explicit(self):
        w=self.workload();h=HNSW(w,m=4,ef_construction=20)
        got,stats=h.search(0,3,32,1,4,adaptive=True)
        self.assertEqual(len(got),3);self.assertLessEqual(stats['final_pool'],4)
        self.assertGreater(stats['search_calls'],1)

    def test_invalid_vectors_rejected(self):
        w=self.workload();w.vectors[0,0]=np.nan
        with self.assertRaises(ValueError):w.validate()

    def test_collapse_requires_vectors_and_eligibility_identical(self):
        w=self.workload()
        self.assertIsNone(collapse_identical_groups(w))
        w.vectors[1]=w.vectors[0]
        self.assertIsNone(collapse_identical_groups(w))
        w.allowed[:,1]=w.allowed[:,0]
        compact,reps=collapse_identical_groups(w)
        self.assertEqual(len(compact.vectors),3)
        for qi in range(2):
            self.assertEqual(set(w.groups[reps[exact(compact,qi,3)]]),set(w.groups[exact(w,qi,3)]))

    def test_vector_readers_variable_ground_truth_length(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.fvecs';a=np.array([[1.,2.],[3.,4.]],dtype='<f4')
            raw=np.zeros((2,3),dtype='<i4');raw[:,0]=2;raw[:,1:]=a.view('<i4');raw.tofile(p)
            np.testing.assert_array_equal(read_fvecs(p,1),a[:1])
            p=Path(d)/'y.ivecs';np.array([2,7,8,0,1,9],dtype='<i4').tofile(p)
            self.assertEqual([x.tolist() for x in read_ivecs(p)],[[7,8],[],[9]])

    def test_benchmark_smoke_schema_and_constraints(self):
        r=benchmark(self.workload(),k=2,efs=[8],pool=2,cap=4,repeats=1)
        self.assertEqual(len(r['summary']),5)
        self.assertEqual(r['summary']['exact_faiss']['group_recall'],1)
        self.assertEqual(r['summary']['exact_prefilter']['group_recall'],1)
        self.assertTrue(all(v['eligibility_violations']==v['group_violations']==0 for v in r['summary'].values()))


if __name__=='__main__':unittest.main()
