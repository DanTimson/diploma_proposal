import tempfile
import unittest
import numpy as np
from diploma_starters.frontier_screen import fresh_data, measure_graph, array_hash
from diploma_starters.resume_graph import Graph
from diploma_starters.group_screen import NativeExact, eligibility


class Frontier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        # Connected bidirectional chain with ties and repeated groups.
        cls.x=np.array([[0.],[1.],[-1.],[2.],[-2.],[3.]],dtype='f4')
        cls.g=np.array([0,0,1,1,2,2],dtype='i8')
        cls.edges=np.array([[-1,1],[0,2],[1,3],[2,4],[3,5],[4,-1]])
        cls.graph=Graph(cls.x,cls.edges,cls.g,cache=cls.tmp.name)
        cls.exact=NativeExact(cls.x,cls.g,cls.tmp.name)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_resume_matches_restart_and_avoids_repeated_distances(self):
        args=(self.graph,np.array([0.],dtype='f4'),np.ones(6,bool),3,[1,3,6],6,0,True)
        a,_,sa=measure_graph(*args,restart=False);b,_,sb=measure_graph(*args,restart=True)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(sa[-1]['distance_evaluations'],6)
        self.assertGreater(sb[-1]['distance_evaluations'],6)

    def test_full_traversal_matches_exact_every_mask(self):
        with self.graph.session(np.array([0.],dtype='f4')) as session:
            session.advance(6)
            for bits in range(64):
                mask=np.array([bool(bits&(1<<i)) for i in range(6)])
                ref=self.exact.search(np.array([0.],dtype='f4'),mask,3,True)
                for cap in [0,1,4]:
                    np.testing.assert_array_equal(session.select(mask,3,6,cap,True),ref)

    def test_cap_before_filter_can_hide_eligible_representative(self):
        mask=np.array([False,True,True,True,True,True])
        with self.graph.session(np.array([0.],dtype='f4')) as s:
            s.advance(6)
            self.assertNotIn(1,s.select(mask,3,6,1,False))
            self.assertIn(1,s.select(mask,3,6,4,False))
            self.assertIn(1,s.select(mask,3,6,1,True))

    def test_disconnected_graph_is_not_falsely_exhaustive(self):
        graph=Graph(self.x,np.full((6,1),-1),self.g,cache=self.tmp.name)
        with graph.session(np.array([0.],dtype='f4')) as s:
            s.advance(6)
            self.assertEqual(s.stats()['expanded'],1)
            self.assertEqual(len(s.select(np.ones(6,bool),3,6)),1)

    def test_invalid_edges_and_closed_or_decreasing_session(self):
        with self.assertRaises(ValueError):Graph(self.x,np.full((6,1),6),self.g,cache=self.tmp.name)
        with self.graph.session(np.array([0.],dtype='f4')) as s:
            s.advance(3)
            with self.assertRaises(ValueError):s.advance(1)
        with self.assertRaises(ValueError):s.advance(6)

    def test_query_count_does_not_change_database(self):
        a,old=fresh_data('clustered',100,64,8,20261115,20261116,'.')
        b,_=fresh_data('clustered',100,64,16,20261115,20261116,'.')
        for key in ['vectors','groups','ranks','base_ids']:self.assertEqual(array_hash(a[key]),array_hash(b[key]))
        self.assertFalse(set(old)&set(a['query_ids']))
        np.testing.assert_array_equal(a['buckets'],b['buckets'][:8])
        self.assertTrue(np.all(eligibility(a,.01)<=eligibility(a,.1)))


if __name__=='__main__':unittest.main()
