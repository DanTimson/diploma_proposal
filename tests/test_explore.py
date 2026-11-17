import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from diploma_starters.explore import (Kernel, metadata, split_ids, read_diskann,
    select_candidates, replay, conditions, summarize, checkpoint, frontier)
from diploma_starters.group_screen import NativeExact


class Explore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cache = cls.temp.name

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_category_sets_and_cross_group_date_filter(self):
        rows = [dict(sub_categories=s, number_of_sub_categories=len(s), update_date=t)
                for s,t in [(['a','b'],1),(['b','a'],9),(['c'],2)]]
        groups,counts,dates = metadata(rows)
        self.assertEqual(groups[0],groups[1])
        data = dict(groups=groups,counts=counts,dates=dates,labels=np.array([2]))
        cond = {name:(mask,grouped) for name,mask,grouped in conditions(data,0,{'0.5':2})}
        self.assertEqual(cond['em_group'][0].tolist(),[True,True,False])
        self.assertEqual(cond['date0.5_group'][0].tolist(),[True,False,True])

    def test_frozen_disjoint_split(self):
        a,b = split_ids(100,7,16); c,d = split_ids(100,7,16)
        np.testing.assert_array_equal(a,c); np.testing.assert_array_equal(b,d)
        self.assertFalse(set(a)&set(b))
        with self.assertRaises(ValueError):split_ids(10,7,6)

    def test_exhaustive_complete_graph_matches_independent_selection(self):
        rng = np.random.default_rng(12); x = rng.normal(size=(31,5)).astype('f4')
        edges = np.tile(np.arange(len(x)),(len(x),1))
        g = rng.integers(0,8,size=len(x)); mask = rng.random(len(x))<.65
        kernel = Kernel(x,edges,self.cache)
        for cap,reserve in [(0,0),(1,0),(4,0),(1,4)]:
            q = rng.normal(size=5).astype('f4')
            got,stats = kernel.search(q,g,mask,[0],len(x),16,cap,reserve,5)
            expected = select_candidates(x,q,g,mask,np.arange(len(x)),5)
            np.testing.assert_array_equal(got,expected)
            self.assertEqual(stats['distances'],len(x))
            self.assertLessEqual(stats['peak_pending'],16)

    def test_ineligible_bridge_and_representative_improvement(self):
        x = np.array([[10.],[8.],[3.],[1.]],dtype='f4')
        edges = np.array([[1,-1],[2,3],[-1,-1],[-1,-1]])
        kernel = Kernel(x,edges,self.cache)
        got,stats = kernel.search([0.],[0,1,0,0],[1,0,1,1],[0],4,8,0,0,2)
        self.assertEqual(got.tolist(),[3])
        self.assertEqual(stats['representative_improvements'],2)
        self.assertEqual(stats['ineligible_discovered'],1)

    def test_distance_budget_includes_entries_and_limits_neighbor_scan(self):
        x = np.arange(20,dtype='f4')[:,None]
        kernel = Kernel(x,np.tile(np.arange(20),(20,1)),self.cache)
        got,stats = kernel.search([0.],np.arange(20),np.ones(20),[19,18,17,16],5,8,0,0,10)
        self.assertEqual(stats['distances'],5)
        self.assertEqual(len(got),5)
        with self.assertRaises(ValueError):
            kernel.search([0.],np.arange(20),np.ones(20),[19,18,17,16],3,8,0,0,10)

    def test_cap_can_remove_a_necessary_route_and_reserve_can_restore_it(self):
        x = np.array([[10.],[1.],[2.],[0.]],dtype='f4')
        edges = np.array([[1,2],[-1,-1],[3,-1],[-1,-1]])
        kernel = Kernel(x,edges,self.cache)
        args = ([0.],[0,0,0,1],[1]*4,[0],20,8)
        capped,_ = kernel.search(*args,1,0,2)
        reserved,stats = kernel.search(*args,1,4,2)
        ordinary,_ = kernel.search(*args,0,0,2)
        self.assertEqual(capped.tolist(),[1])
        self.assertEqual(reserved.tolist(),[3,1])
        np.testing.assert_array_equal(ordinary,reserved)
        self.assertLessEqual(stats['peak_pending'],8)

    def test_empty_filter_ties_and_invalid_inputs(self):
        x = np.zeros((4,2),dtype='f4'); edges = np.tile(np.arange(4),(4,1))
        kernel = Kernel(x,edges,self.cache)
        got,_ = kernel.search([0.,0.],[0,0,1,1],[0,0,0,0],[3],4,8,0,0,5)
        self.assertEqual(len(got),0)
        got,_ = kernel.search([0.,0.],[0,0,1,1],[1,1,1,1],[3],4,8,0,0,5)
        self.assertEqual(got.tolist(),[0,2])
        with self.assertRaises(ValueError):
            kernel.search([float('nan'),0.],[0,0,1,1],[1]*4,[0],4,8,0,0,5)
        with self.assertRaises(ValueError):Kernel(x,np.full((4,1),4),self.cache)

    def test_diskann_format_rejects_truncation_and_frozen_nodes(self):
        path = Path(self.cache)/'graph'
        payload = struct.pack('<QIIQ',48,2,1,0)+struct.pack('<III',2,1,2)+struct.pack('<II',1,0)+struct.pack('<I',0)
        path.write_bytes(payload)
        edges,entry = read_diskann(path,3)
        self.assertEqual(entry,1); self.assertEqual(edges.tolist(),[[1,2],[0,-1],[-1,-1]])
        path.write_bytes(payload[:-1])
        with self.assertRaises(ValueError):read_diskann(path,3)
        path.write_bytes(struct.pack('<QIIQ',48,2,1,1)+payload[24:])
        with self.assertRaises(ValueError):read_diskann(path,3)

    def test_replay_reports_ties_but_rejects_wrong_ground_truth(self):
        x = np.array([[1.],[-1.],[5.]],dtype='f4')
        exact = NativeExact(x,np.arange(3),self.cache)
        result = replay(x,np.array([0.],dtype='f4'),np.ones(3,bool),[1],exact,1)
        self.assertFalse(result['id_set_equal'])
        self.assertTrue(result['distances_within_tolerance'])
        with self.assertRaises(ValueError):
            replay(x,np.array([0.],dtype='f4'),np.ones(3,bool),[2],exact,1)

    def test_checkpoint_detects_changed_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'graph.npy'; path.write_bytes(b'first')
            checkpoint(root,'graph',[path],{'entry':0})
            self.assertEqual(checkpoint(root,'graph',[]),{'entry':0})
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):checkpoint(root,'graph',[])

    def test_frontier_requires_representatives_and_feasibility(self):
        row = dict(condition='group',k=10,method='a',queries=4,feasible_queries=4,
                   violations=0,item_recall=.9,group_recall=1.,fill=1.,ms=1.,budget=32)
        self.assertIsNone(frontier([row])[0]['best_at_mean_recall_95'])
        row['item_recall'] = .98; row['feasible_queries'] = 3
        self.assertIsNone(frontier([row])[0]['best_at_mean_recall_95'])
        row['feasible_queries'] = 4
        self.assertEqual(frontier([row])[0]['best_at_mean_recall_95']['budget'],32)


if __name__ == '__main__':unittest.main()
