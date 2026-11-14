import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from diploma_starters.metrics import note_metrics,pair_metrics,target_hit_rate
from diploma_starters.prototypes import aggregate,history_cap_weights,select_evidence
from diploma_starters.assets import extract_scifact


class Metrics(unittest.TestCase):
    def test_different_error_denominators(self):
        r=note_metrics([1,1,0,0,0,0],[1,0,1,1,0,0])
        self.assertEqual(r['suppression'],2/3)
        self.assertEqual(r['waste'],2/4)
        self.assertEqual(r['pollution'],1/2)
        self.assertIsNone(note_metrics([0],[1])['pollution'])

    def test_wrong_stance_is_not_true_positive(self):
        r=pair_metrics({(1,2):'SUPPORT'},{(1,2):'CONTRADICT'})
        self.assertEqual(r['f1'],0)

    def test_missing_evidence_penalizes_recall(self):
        r=pair_metrics({(1,2):'SUPPORT',(1,3):'CONTRADICT'},{(1,2):'SUPPORT'})
        self.assertEqual(r['recall'],.5);self.assertEqual(r['precision'],1)

    def test_unjudged_is_fp_in_official_protocol(self):
        r=pair_metrics({(1,2):'SUPPORT'},{(1,2):'SUPPORT',(1,3):'SUPPORT'})
        self.assertEqual(r['precision'],.5)

    def test_seen_target_excluded_seen_items_masked(self):
        r=target_hit_rate({1:[9,2],2:[9,2]},2,{1:[2],2:[9]},1)
        self.assertEqual(r,{'eligible_users':1,'target_hr':1.0})

    def test_clone_mass_invariant(self):
        one={'a':[(1,1)],'b':[(2,1)]}
        many={**one,**{f'c{i}':[(2,1)] for i in range(10)}}
        baseline=aggregate([('a',1),('b',0)],history_cap_weights(one))
        attacked=aggregate([('a',1),('b',0),*[(f'c{i}',0) for i in range(10)]],history_cap_weights(many))
        self.assertAlmostEqual(baseline,attacked)

    def test_diversified_histories_escape_exact_cap(self):
        w=history_cap_weights({f'a{i}':[(i,1)] for i in range(10)})
        self.assertEqual(sum(w.values()),10)

    def test_cold_start_not_protected(self):
        self.assertEqual(sum(history_cap_weights({'a':[],'b':[]}).values()),2)

    def test_gold_labels_do_not_affect_selection(self):
        c=[dict(id='a',origin='x',score=.9,predicted_stance='SUPPORT',gold='SUPPORT'),
           dict(id='b',origin='x',score=.8,predicted_stance='SUPPORT',gold='CONTRADICT'),
           dict(id='c',origin='y',score=.7,predicted_stance='CONTRADICT',gold='NEI')]
        ids=[x['id'] for x in select_evidence(c,2)]
        for x in c:x['gold']='OTHER'
        self.assertEqual(ids,[x['id'] for x in select_evidence(c,2)])
        self.assertEqual(ids,['a','c'])

    def test_duplicate_target_rating_rejected(self):
        with self.assertRaises(ValueError):aggregate([('a',0),('a',1)])

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.tar.gz'
            with tarfile.open(path,'w:gz') as t:
                m=tarfile.TarInfo('../escape');m.size=1;t.addfile(m,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):extract_scifact(path,Path(d)/'out')


if __name__=='__main__':unittest.main()
