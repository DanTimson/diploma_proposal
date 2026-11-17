from pathlib import Path
import tempfile
import unittest
import numpy as np
import faiss
from diploma_starters.baseline_controls import (SIMDExact,upper_entry,diagnostic,
    faiss_candidates,numerical_audit,summarize)
from diploma_starters.explore import Kernel
from diploma_starters.group_screen import NativeExact,write_bin
from diploma_starters.upstream_controls import instrument,recipes,read_probe,verify_stored_vectors


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.cache=cls.temp.name
        faiss.omp_set_num_threads(1)
        cls.x=np.random.default_rng(23).normal(size=(150,32)).astype('f4')
        cls.groups=np.arange(150,dtype='i8')//3
        cls.index=faiss.IndexHNSWFlat(32,8);cls.index.add(cls.x)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_simd_matches_reference_with_filters_and_groups(self):
        scalar=NativeExact(self.x,self.groups,self.cache)
        fast=SIMDExact(self.x,self.groups,self.cache)
        rng=np.random.default_rng(37)
        for grouped in (False,True):
            for fraction in (0,.1,.5,1):
                mask=rng.random(len(self.x))<fraction
                for k in (1,10,100):
                    a=scalar.search(self.x[17],mask,k,grouped)
                    b=fast.search(self.x[17],mask,k,grouped)
                    np.testing.assert_array_equal(a,b)

    def test_simd_tie_break_and_numerical_audit(self):
        x=np.array([[1.],[-1.],[5.]],dtype='f4')
        fast=SIMDExact(x,np.array([0,0,1]),self.cache)
        self.assertEqual(fast.search([0.],[1,1,1],2,True).tolist(),[0,2])
        self.assertEqual(numerical_audit(x,np.array([0.]),np.array([1]),np.array([0])),
                         'different_ids_with_near_equal_distances')
        with self.assertRaises(ValueError):
            numerical_audit(x,np.array([0.]),np.array([2]),np.array([0]))

    def test_upper_descent_matches_independent_neighbor_walk(self):
        h=self.index.hnsw;offsets=faiss.vector_to_array(h.offsets)
        edges=faiss.vector_to_array(h.neighbors);cum=faiss.vector_to_array(h.cum_nneighbor_per_level)
        for q in self.x[[3,40,110]]:
            dc=self.index.storage.get_distance_computer();dc.set_query(faiss.swig_ptr(q))
            node=int(h.entry_point);distance=dc(node);calls=1
            for level in range(h.max_level,0,-1):
                while True:
                    old=node
                    begin=int(offsets[old]+cum[level]);end=int(offsets[old]+cum[level+1])
                    for neighbor in edges[begin:end]:
                        if neighbor<0:break
                        calls+=1;value=dc(int(neighbor))
                        if value<distance:node=int(neighbor);distance=value
                    if node==old:break
            actual,stats=upper_entry(self.index,q)
            self.assertEqual(actual,node);self.assertEqual(stats['upper_distances'],calls)

    def test_hierarchy_charges_upper_distances_to_total_budget(self):
        h=self.index.hnsw;off=faiss.vector_to_array(h.offsets);neighbors=faiss.vector_to_array(h.neighbors)
        edges=np.array([neighbors[int(o):int(o)+h.nb_neighbors(0)] for o in off[:-1]],dtype='i8')
        kernel=Kernel(self.x,edges,self.cache)
        got,stats=diagnostic(kernel,self.index,self.x[3],self.groups,np.ones(150,bool),
                             [h.entry_point],10,256,'hierarchical')
        self.assertGreater(stats['upper_distances'],0)
        self.assertEqual(stats['total_distances'],stats['upper_distances']+stats['distances'])
        self.assertLessEqual(stats['total_distances'],256)
        with self.assertRaises(ValueError):
            diagnostic(kernel,self.index,self.x[3],self.groups,np.ones(150,bool),[h.entry_point],10,1,'hierarchical')

    def test_faiss_settings_are_independent_and_filter_enforced(self):
        mask=np.arange(150)%2==0
        for ef in (32,128):
            ids,stats=faiss_candidates(self.index,self.x[7],mask,ef,16,True)
            self.assertEqual(stats['ef_search'],ef);self.assertEqual(stats['pool'],16)
            self.assertTrue(mask[ids[ids>=0]].all());self.assertGreater(stats['search_distances'],0)

    def test_instrumentation_changes_only_clock_and_logging(self):
        anchor='        std::sort(latency_stats.begin(), latency_stats.end());'
        original='auto start = std::chrono::high_resolution_clock::now();\n'+anchor+'\nauto end = std::chrono::high_resolution_clock::now();'
        altered,addition=instrument(original)
        self.assertEqual(altered.replace(addition,''),original.replace('std::chrono::high_resolution_clock','std::chrono::steady_clock'))
        self.assertNotIn('high_resolution_clock',altered)
        self.assertEqual(altered.count('steady_clock::now()'),2)
        self.assertLess(altered.index('_per_query.tsv'),altered.index('std::sort'))
        with self.assertRaises(ValueError):instrument(original+original)

    def test_reuses_build_flags_without_overwriting_original_target(self):
        flags='CXX_DEFINES = -DUSE_AVX2\nCXX_INCLUDES = -I/x/include\nCXX_FLAGS = -O3 -fopenmp\n'
        link='/usr/bin/c++ -O3 CMakeFiles/search_memory_index.dir/search_memory_index.cpp.o -o search_memory_index ../src/libdiskann.so'
        compile_cmd,link_cmd=recipes(flags,link,'/tmp/probe.cpp','/tmp/probe.o','/tmp/probe')
        self.assertIn('-DUSE_AVX2',compile_cmd)
        self.assertEqual(link_cmd[link_cmd.index('-o')+1],'/tmp/probe')
        self.assertNotIn('CMakeFiles/search_memory_index.dir/search_memory_index.cpp.o',link_cmd)
        with self.assertRaises(ValueError):recipes(flags,link+' && bad','a','b','c')

    def test_probe_reader_preserves_query_order_and_missing_sentinel(self):
        prefix=Path(self.cache)/'probe'
        path=Path(str(prefix)+'_32_idx_uint32.bin')
        with path.open('wb') as f:
            np.array([2,2,0,2**32-1,1,2],dtype='<u4').tofile(f)
        timing=Path(str(prefix)+'_32_per_query.tsv');timing.write_text('0\t1000\t12\n1\t2000\t20\n')
        ids,ms,counts=read_probe(prefix,32,2,2,3)
        self.assertEqual(ids.tolist(),[[0,-1],[1,2]]);self.assertEqual(ms.tolist(),[1.,2.])
        timing.write_text('1\t1000\t12\n0\t2000\t20\n')
        with self.assertRaises(ValueError):read_probe(prefix,32,2,2,3)

    def test_upstream_stored_vectors_must_match_prepared_vectors(self):
        path=Path(self.cache)/'vectors.bin';write_bin(path,self.x)
        verify_stored_vectors(path,self.x)
        with self.assertRaises(ValueError):verify_stored_vectors(path,self.x+1)


if __name__=='__main__':unittest.main()
