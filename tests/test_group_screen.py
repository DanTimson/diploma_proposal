import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
from diploma_starters.group_screen import NativeExact, evaluate, eligibility, make_data

class GroupScreen(unittest.TestCase):
    def test_native_matches_exhaustive_constrained_sets(self):
        # Independent combinatorial oracle, integer coordinates avoid tolerance ambiguity.
        x=np.array([[0,0],[1,0],[2,0],[0,3],[4,0],[0,5]],dtype='f4');g=np.array([0,0,1,2,1,3],dtype='i8');q=np.array([0,0],dtype='f4')
        with tempfile.TemporaryDirectory() as tmp:
            ex=NativeExact(x,g,tmp)
            for mask_bits in range(64):
                mask=np.array([(mask_bits>>i)&1 for i in range(6)],bool)
                for grouped in [False,True]:
                    got=ex.search(q,mask,3,grouped)
                    candidates=[]
                    for n in range(4):
                        for s in itertools.combinations(np.flatnonzero(mask),n):
                            if grouped and len(set(g[list(s)]))<n:continue
                            candidates.append((-n,sum(float(x[i]@x[i]) for i in s),s))
                    expected=min(candidates)[2]
                    self.assertEqual(set(got),set(expected))
    def test_equal_distance_stable_and_empty(self):
        x=np.array([[1,0],[-1,0],[0,1]],dtype='f4');g=np.array([0,0,1],dtype='i8')
        with tempfile.TemporaryDirectory() as tmp:
            ex=NativeExact(x,g,tmp)
            self.assertEqual(ex.search(np.zeros(2,'f4'),np.ones(3,bool),2,True).tolist(),[0,2])
            self.assertEqual(len(ex.search(np.zeros(2,'f4'),np.zeros(3,bool),2,True)),0)
    def test_native_rejects_unsafe_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):NativeExact(np.zeros((3,2),'f4'),np.zeros(2,'i8'),tmp)
            ex=NativeExact(np.zeros((3,2),'f4'),np.arange(3,dtype='i8'),tmp)
            with self.assertRaises(ValueError):ex.search(np.zeros(1,'f4'),np.ones(3,bool),1,True)
    def test_group_recall_does_not_hide_bad_representative(self):
        x=np.array([[0],[10],[1]],'f4');q=np.array([0],'f4');g=np.array([0,0,1]);mask=np.ones(3,bool)
        r=evaluate(x,q,g,mask,[1,2],[0,2],True)
        self.assertEqual(r['group_recall'],1);self.assertEqual(r['item_recall'],.5)
        self.assertGreater(r['mean_l2_regret'],0)
    def test_factorial_masks_nested_and_data_fixed(self):
        a=make_data('clustered',1000,8,11,None);b=make_data('clustered',1000,8,11,None)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        small,large=eligibility(a,.01),eligibility(a,.1)
        self.assertTrue(np.all(~small|large));self.assertTrue(np.all(small.sum(axis=1)==10))
    def test_invalid_duplicate_and_ineligible_are_reported(self):
        r=evaluate(np.array([[0],[1]],'f4'),np.zeros(1,'f4'),np.array([0,0]),np.array([True,False]),[1,1,99],[0],True)
        self.assertEqual((r['invalid_ids'],r['duplicate_ids'],r['eligibility_violations'],r['group_violations']),(1,1,2,1))
if __name__=='__main__':unittest.main()
