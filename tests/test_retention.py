import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from diploma_starters.retention import FastSelector,RetentionKernel
from diploma_starters.explore import select_candidates
from diploma_starters.d008_decision import quality,paired,decide,BUDGETS
from diploma_starters.explore import identity,atomic


class Retention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.temp=tempfile.TemporaryDirectory()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_fast_selection_matches_reference_and_rejects_invalid_ids(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(70,32)).astype('f4');g=np.arange(70)//3
        fast=FastSelector(x,g,self.temp.name);q=x[10];mask=rng.random(70)<.7
        ids=np.r_[rng.integers(0,70,90),-1,2**32-1]
        for k in (1,10,100):
            np.testing.assert_array_equal(fast.select(q,mask,ids,k),select_candidates(x,q,g,mask,ids,k))
        with self.assertRaises(ValueError):fast.select(q,mask,[70],10)

    def test_filter_before_group_preserves_eligible_alternative_without_changing_route(self):
        x=np.array([[0.],[1.],[2.]],dtype='f4');g=np.array([0,0,1]);edges=np.tile(np.arange(3),(3,1))
        kernel=RetentionKernel(x,edges,self.temp.name)
        before,a=kernel.search([0.],g,[0,1,1],[0],3,8,0,0,2)
        after,b=kernel.search([0.],g,[0,1,1],[0],3,8,2,0,2)
        self.assertEqual(before.tolist(),[1,2]);self.assertEqual(after.tolist(),[2])
        for field in ('distances','expansions','peak_pending'):self.assertEqual(a[field],b[field])

    def test_alternation_keeps_ineligible_routing_and_has_shared_pending_bound(self):
        # Entry is ineligible; it exposes nearer ineligible and further eligible paths.
        x=np.arange(8,dtype='f4')[:,None];edges=np.array([[1,4],[2,-1],[3,-1],[-1,-1],[5,-1],[6,-1],[7,-1],[-1,-1]])
        g=np.arange(8);mask=np.array([0,0,0,1,1,1,1,1])
        kernel=RetentionKernel(x,edges,self.temp.name)
        for mode in (0,1,2,3):
            got,stats=kernel.search([0.],g,mask,[0],8,8,mode,0,5)
            self.assertTrue(mask[got].all());self.assertLessEqual(stats['peak_pending'],8)
            self.assertEqual(stats['distances'],8);self.assertIn(3,got)

    def test_candidate_policy_can_change_early_discoveries_at_equal_budget(self):
        x=np.arange(8,dtype='f4')[:,None];edges=np.array([[1,4],[2,-1],[3,-1],[-1,-1],[5,-1],[6,-1],[7,-1],[-1,-1]])
        kernel=RetentionKernel(x,edges,self.temp.name);g=np.arange(8);mask=[0,0,0,1,1,1,1,1]
        a,sa=kernel.search([0.],g,mask,[0],4,8,0,0,5)
        b,sb=kernel.search([0.],g,mask,[0],4,8,1,0,5)
        self.assertEqual(a.tolist(),[4]);self.assertEqual(b.tolist(),[4,5])
        self.assertEqual(sa['distances'],sb['distances'])

    def test_gate_requires_quality_and_material_paired_speedup(self):
        rows=[dict(query_id=i,feasible=True,item_recall=.96,fill=1.,ms=10.,
                   eligibility_violations=0,group_violations=0,invalid_ids=0,duplicate_ids=0) for i in range(16)]
        self.assertTrue(quality(rows))
        baselines=[dict(r,ms=13.) for r in rows]
        result=paired(rows,baselines)
        self.assertAlmostEqual(result['geomean_speedup'],1.3)
        self.assertEqual(result['faster_queries'],16);self.assertGreater(result['bootstrap_lower95'],1.)
        rows[0]['item_recall']=.79;self.assertFalse(quality(rows))
        with self.assertRaises(ValueError):paired(rows,baselines[:-1])

    def test_full_gate_pass_stop_and_incomplete_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=dict(smoke=False,queries=16,query_ids=list(range(16)),ks=[10,100],repeats=3,
                      upstream={'verified':True},budgets=list(BUDGETS),grid=[[256,256]],upstream_grid=[[256,256]])
            atomic(root/'plan.json',plan);atomic(root/'summary.json',{'status':'complete'})
            names=['exact_reference','exact_simd','faiss_a','faiss_b']
            names += [f'hnsw0_{mode}/D{budget}' for budget in BUDGETS for mode in
                      ('ordinary_before','ordinary_after','alternate_before','alternate_after')]
            names += [f'upstream_{i}' for i in range(6)]
            files=[]
            for qi in range(16):
                for condition in ('group_all','em_group','date0.1_group','date0.5_group'):
                    for k in (10,100):
                        rows=[dict(query_id=qi,condition=condition,k=k,method=name,feasible=True,
                            item_recall=1.,fill=1.,ms=7. if 'alternate_before' in name else 10.,
                            timing_scope='integrated_query',eligibility_violations=0,group_violations=0,
                            invalid_ids=0,duplicate_ids=0) for name in names]
                        path=root/'cells'/f'{qi}-{condition}-{k}.json';files.append(path)
                        atomic(path,dict(plan=identity(plan),rows=rows,checksum=identity(rows)))
            report=decide(root)
            self.assertEqual(report['verdict'],'CONFIRM_ONE_FROZEN_POLICY')
            self.assertEqual(report['selected']['method'],'hnsw0_alternate_before/D1024')
            for path in files:
                cell=json.loads(path.read_text())
                for row in cell['rows']:
                    if 'alternate_before' in row['method']:row['ms']=9.
                cell['checksum']=identity(cell['rows']);atomic(path,cell)
            self.assertEqual(decide(root)['verdict'],'STOP_D_AND_SWITCH_TO_A_B')
            files[0].unlink()
            with self.assertRaises(ValueError):decide(root)


if __name__=='__main__':unittest.main()
