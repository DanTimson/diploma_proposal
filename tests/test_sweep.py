import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

LINUX=sys.platform.startswith('linux')
ANN=all(importlib.util.find_spec(x) is not None for x in ['numpy','faiss','scipy','sklearn'])
if LINUX:
    from diploma_starters import sweep


@unittest.skipUnless(LINUX,'Sweep resource/process controls require Linux/WSL')
class Sweep(unittest.TestCase):
    def plan(self):
        return dict(schema=1,hours=.02,job_minutes=1,memory_gib=4,jobs=[])

    def test_plan_rejects_duplicate_ids(self):
        p=self.plan();p['jobs']=[dict(id='x',kind='fetch')]*2
        with self.assertRaises(ValueError):sweep.validate_plan(p)

    def test_plan_rejects_forward_dependencies(self):
        p=self.plan();p['jobs']=[dict(id='x',kind='fetch',requires=['missing'])]
        with self.assertRaises(ValueError):sweep.validate_plan(p)

    def test_atomic_json_and_receipt_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.json';sweep.atomic_json(p,dict(receipt_token='abc',summary={'x':1}))
            digest=sweep.sha256(p)
            self.assertTrue(sweep.receipt_ok(p,'abc',digest))
            self.assertFalse(sweep.receipt_ok(p,'wrong',digest))
            p.write_text('{broken')
            self.assertFalse(sweep.receipt_ok(p,'abc',digest))

    def test_timeout_terminates_child(self):
        with tempfile.TemporaryDirectory() as d:
            status,_,elapsed=sweep.run_child([sys.executable,'-c','import time; time.sleep(5)'],Path(d)/'log',d,.1,2,lambda:False)
            self.assertEqual(status,'timeout');self.assertLess(elapsed,4)

    def test_cancel_terminates_child(self):
        with tempfile.TemporaryDirectory() as d:
            status,_,_=sweep.run_child([sys.executable,'-c','import time; time.sleep(5)'],Path(d)/'log',d,5,2,lambda:True)
            self.assertEqual(status,'interrupted')

    def test_stop_refuses_reused_pid(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            sweep.atomic_json(Path(d)/'pid.json',dict(pid=os.getpid(),start_ticks='not-the-current-process'))
            self.assertIn('nothing signalled',sweep.stop_run(d))

    def test_child_memory_limit_is_applied(self):
        with tempfile.TemporaryDirectory() as d:
            code='import resource; assert resource.getrlimit(resource.RLIMIT_AS)[0] == 2*1024**3'
            status,_,_=sweep.run_child([sys.executable,'-c',code],Path(d)/'log',d,5,2,lambda:False)
            self.assertEqual(status,'complete')

    @unittest.skipUnless(ANN,'ANN dependencies needed for integration')
    def test_failure_isolated_success_resumed_and_plan_guarded(self):
        p=self.plan();args=['--queries','2','--efs','8','--pool','10','--cap','20','--repeats','1','synthetic','--n','80']
        p['jobs']=[dict(id='bad',kind='benchmark',args=['--invalid-argument']),
                   dict(id='blocked',kind='benchmark',requires=['bad'],args=args),
                   dict(id='good',kind='benchmark',args=args)]
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            state=sweep.execute(repo,p,d)
            self.assertEqual(state['jobs']['bad']['status'],'failed')
            self.assertEqual(state['jobs']['blocked']['status'],'blocked')
            self.assertEqual(state['jobs']['good']['status'],'complete')
            result=Path(d)/'results/good.json';before=result.stat().st_mtime_ns
            sweep.execute(repo,p,d)
            self.assertEqual(before,result.stat().st_mtime_ns)
            changed=dict(p,hours=.03)
            with self.assertRaises(ValueError):sweep.execute(repo,changed,d)


@unittest.skipUnless(ANN,'ANN dependencies required')
class BlockedReference(unittest.TestCase):
    def test_blocking_preserves_exact_selection(self):
        import numpy as np
        from diploma_starters.ann import Workload,exact,select,scores
        rng=np.random.default_rng(19)
        for metric in ['ip','l2']:
            w=Workload('block',rng.normal(size=(77,8)),rng.normal(size=(3,8)),
                       rng.integers(0,20,size=77),rng.random((3,77))>.4,metric).validate()
            for qi in range(3):
                ids=np.flatnonzero(w.allowed[qi]);v=scores(w.vectors[ids],w.queries[qi],metric)
                expected=select(ids,v,w.groups,w.allowed[qi],10)
                np.testing.assert_array_equal(exact(w,qi,10,block_size=7),expected)


if __name__=='__main__':unittest.main()
