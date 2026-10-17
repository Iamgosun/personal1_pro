import csv
import json
import tempfile
import unittest
from pathlib import Path
from experiments.supplementary.reporting import paired_e5, make_tables, aggregate_seeds


def row(method, budget=60., ref=512, test=10, digest='same-input', repeats=2, elapsed=60.1):
    return dict(dataset='caltech101',shot=16,seed=1,method=method,
        budget_seconds=budget,train_seconds=elapsed,updates=100,
        budget_reached=True,last_step_seconds=1.,s_ref=ref,n_mc_test=test,
        mc_repeats=repeats,native_max_epoch=50,batch_size=32,reference_split='test',formal_split='test',
        reference_input_sha256=digest,branch_nll_ref=.3,original_nll_ref=.2,
        fusion_acc=90.,fusion_ece=1.,ambiguous_reference=False)


class Reporting(unittest.TestCase):
    def test_e5_pair_requires_equal_budget(self):
        yes,no=paired_e5([row('branch',budget=60),row('davi',budget=90,elapsed=90.1)])
        self.assertEqual(len(yes),0)
        self.assertIn('budgets',no[0]['reasons'])

    def test_e5_pair_requires_equal_sampling(self):
        yes,no=paired_e5([row('branch',ref=256),row('davi',ref=512)])
        self.assertEqual(len(yes),0)
        self.assertIn('MC sample',no[0]['reasons'])

    def test_e5_pair_requires_paired_inputs(self):
        yes,no=paired_e5([row('branch',digest='a'),row('davi',digest='b')])
        self.assertEqual(len(yes),0)
        self.assertIn('digests',no[0]['reasons'])

    def test_e5_pair_accepts_only_valid_pair(self):
        yes,no=paired_e5([row('branch'),row('davi')])
        self.assertEqual(len(yes),2)
        self.assertEqual(len(no),0)

    def test_e5_rejects_ambiguous_and_unfinished(self):
        x=row('branch');x['ambiguous_reference']=True
        y=row('davi');y['budget_reached']=False
        yes,no=paired_e5([x,y]);self.assertEqual(len(yes),0)
        self.assertIn('multiple reference',no[0]['reasons'])
        self.assertIn('not reached',no[0]['reasons'])

    def test_e5_rejects_scheduler_and_batch_protocol_mismatch(self):
        x=row('branch');y=row('davi');y['native_max_epoch']=40;y['batch_size']=24
        valid,invalid=paired_e5([x,y])
        self.assertFalse(valid)
        self.assertIn('scheduler',invalid[0]['reasons'])
        self.assertIn('batch sizes',invalid[0]['reasons'])

    def test_seed_mean_sample_std(self):
        result=aggregate_seeds([{'dataset':'x','method':'davi','seed':1,'acc':1.},
                {'dataset':'x','method':'davi','seed':2,'acc':3.}],['dataset','method'],['acc'])
        self.assertEqual(result[0]['acc_mean'],2)
        self.assertAlmostEqual(result[0]['acc_std'],2**.5)

    def test_e5_end_to_end_report_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for method,budget,sref in [('branch',70,256),('davi',90,512)]:
                run=root/'matched_time'/'caltech101'/method/'shots_16'/'seed_1'
                manifest={'method':method,'dataset':'caltech101','shot':16,'seed':1,'status':'complete',
                          'native_max_epoch':50,'batch_size':32}
                info={'budget_seconds':budget,'elapsed_training_seconds':budget+.1,'updates':100,
                      'budget_reached':True,'last_step_seconds':1.,'last_checkpoint_epoch':4}
                formal={'dataset':'caltech101','method':method,'shot':16,'seed':1,'partial':False,
                    'checkpoint':{'selection':'latest','epoch':4},'n_mc_test':10,'fusion_accuracy':90,
                    'main_accuracy':89,'rep_accuracy':85,'rep_predictive_nll':.4,
                    'metrics':{'ece':2.,'aece':2.5}}
                ref={'dataset':'caltech101','method':method,'shot':16,'seed':1,'partial':False,
                    'checkpoint':{'selection':'latest','epoch':4},'s_ref':sref,'mc_repeats':2,
                    'objective_type':'native_avi_mc_only' if method=='branch' else 'native_davi',
                    'input_sha256':'same','original_nll':.2,'branch_nll':.3,'gap1':.1,
                    'r_mean_fro':.1,'r_trace':.02,'t_trace':.03}
                for p,value in ((run/'manifest.json',manifest),(run/'training.json',info),
                    (run/'formal'/'latest'/'epoch_4'/'complete'/'test.json',formal),
                    (run/'diagnostics'/'latest'/'epoch_4'/f'S{sref}_R2'/'test'/'metrics.json',ref)):
                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_text(json.dumps(value),encoding='utf-8')
            output=make_tables(root)
            self.assertEqual(output['e5_accepted_runs'],0)
            self.assertEqual(output['e5_rejected_pairs'],1)
            with (root/'summary'/'table_s4_matched_runs.csv').open(newline='') as f:
                self.assertEqual(list(csv.DictReader(f)),[])
            with (root/'summary'/'table_s4_rejected_pairs.csv').open(newline='') as f:
                self.assertIn('budgets',next(csv.DictReader(f))['reasons'])


if __name__=='__main__':unittest.main()
