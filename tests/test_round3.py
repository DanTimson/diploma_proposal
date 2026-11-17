import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from diploma_starters.group_screen import NativeExact

def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Round3(unittest.TestCase):
    def test_sparse_exact_matches_dense_all_masks_ties_and_unique_groups(self):
        x=np.array([[0,0],[1,0],[-1,0],[2,0],[0,2],[0,-2]],dtype='f4');q=np.zeros(2,dtype='f4')
        with tempfile.TemporaryDirectory() as td:
            for groups in [np.array([0,0,1,1,5,5]),np.arange(6)]:
                engines=[NativeExact(x,groups,td,implementation=kind) for kind in ['dense','sparse']]
                for bits in range(64):
                    mask=np.array([bool(bits&(1<<i)) for i in range(6)])
                    for grouped in [False,True]:
                        np.testing.assert_array_equal(*(e.search(q,mask,3,grouped) for e in engines))

    def test_copy_dedup_invariance_and_slot_crowding(self):
        b=module('evidence_copy_pilot')
        cs=[dict(id=str(i),score=1-i/10,text_hash=str(i),doc_id=i,label='SUPPORT') for i in range(5)]
        clean=b.select(b.perturb(cs,0),3,True);dirty=b.perturb(cs,20)
        self.assertEqual([c['doc_id'] for c in clean],[c['doc_id'] for c in b.select(dirty,3,True)])
        self.assertEqual(len({c['doc_id'] for c in b.select(dirty,3,False)}),1)
        changed=[dict(c,gold_label='CONTRADICT') for c in dirty]
        self.assertEqual([c['id'] for c in b.select(changed,3,True)],[c['id'] for c in b.select(dirty,3,True)])

    def test_category_sets_are_order_invariant(self):
        audit=module('audit_fanns_groups').audit
        rows=[dict(sub_categories=['b','a'],authors=['x','y'],number_of_sub_categories=2),
              dict(sub_categories=['a','b'],authors=['x','z'],number_of_sub_categories=2)]
        r=audit(rows)
        self.assertEqual(r['exact_subcategory_set']['groups'],1)
        self.assertEqual(r['exact_author_set']['groups'],2)
        self.assertEqual(r['repeated_author_strings'],1)
if __name__=='__main__':unittest.main()
