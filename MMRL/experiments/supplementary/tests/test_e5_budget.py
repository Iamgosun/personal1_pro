"""E5 simulated native optimizer/scheduler, timing and evidence lineage; no Dassl/GPU."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from experiments.supplementary.cli import budget_training_updates
from experiments.supplementary.core import sha256
from experiments.supplementary.reporting import make_tables, paired_e5


class FakeTime:
    def __init__(self): self.t=0.0
    def __call__(self): return self.t


class MockNativeTrainer:
    def __init__(self, clock, batches=3, epochs=3):
        self.dm=SimpleNamespace(train_loader_x=list(range(batches)))
        self.cfg=SimpleNamespace(OPTIM=SimpleNamespace(MAX_EPOCH=epochs))
        self.clock=clock
        self.weight=torch.nn.Parameter(torch.tensor([2.0]))
        self.optim=torch.optim.SGD([self.weight],lr=0.1)
        self.epochs=[];self.lr_updates=0;self.after_epoch_calls=0
    def set_model_mode(self, mode): assert mode=='train'
    def before_epoch(self): self.epochs.append(self.epoch)
    def after_epoch(self): self.after_epoch_calls+=1
    def forward_backward(self,batch):
        self.optim.zero_grad()
        loss=(self.weight-(batch+1.)).square().sum()
        loss.backward();self.optim.step()
        # Mirrors OnlineExecutor last-batch scheduler trigger.
        if self.batch_idx+1==self.num_batches:
            self.lr_updates+=1
            for group in self.optim.param_groups:group['lr']*=.5
        self.clock.t+=1.0
        return {'loss':float(loss.detach()),'acc':50.0}


class E5Budget(unittest.TestCase):
    def test_mid_epoch_checkpoint_and_lr_are_precise(self):
        clock=FakeTime();trainer=MockNativeTrainer(clock)
        state=budget_training_updates(trainer,budget_seconds=4.5,clock=clock)
        self.assertEqual(state['updates'],5)
        self.assertEqual(state['completed_epochs'],1)
        self.assertEqual(state['last_checkpoint_epoch'],2)
        self.assertEqual(state['last_update_batch_in_epoch'],2)
        self.assertTrue(state['partial_epoch'])
        self.assertEqual(trainer.lr_updates,1)
        self.assertEqual(trainer.after_epoch_calls,0)
        self.assertAlmostEqual(state['elapsed_training_seconds'],5.0)
        self.assertAlmostEqual(state['overshoot_seconds'],.5)

    def test_exact_epoch_boundary_is_not_partial(self):
        clock=FakeTime();t=MockNativeTrainer(clock)
        state=budget_training_updates(t,budget_seconds=3.0,clock=clock)
        self.assertEqual(state['completed_epochs'],1)
        self.assertFalse(state['partial_epoch'])
        self.assertEqual(state['last_update_batch_in_epoch'],3)
        self.assertEqual(t.lr_updates,1)

    def test_epoch_cap_rejection_without_losing_pilot(self):
        clock=FakeTime();t=MockNativeTrainer(clock,batches=2,epochs=1)
        state=budget_training_updates(t,budget_seconds=9.,clock=clock)
        self.assertFalse(state['budget_reached'])
        self.assertEqual(state['updates'],2)
        self.assertEqual(state['completed_epochs'],1)

    def test_full_epoch_parameter_and_scheduler_parity(self):
        clock=FakeTime();a=MockNativeTrainer(clock)
        # Native reference: TrainerX's per-batch optimizer path under a full epoch.
        b=MockNativeTrainer(FakeTime())
        for trainer in (a,b):
            trainer.weight.data.fill_(2.0)
        recorded=budget_training_updates(a,budget_seconds=3.,clock=clock)
        b.epoch=0;b.max_epoch=3;b.num_batches=3
        b.set_model_mode('train');b.before_epoch()
        for idx,batch in enumerate(b.dm.train_loader_x):
            b.batch_idx=idx;b.forward_backward(batch)
        self.assertTrue(torch.equal(a.weight.detach(),b.weight.detach()))
        self.assertEqual(a.lr_updates,b.lr_updates)
        self.assertEqual(a.optim.param_groups[0]['lr'],b.optim.param_groups[0]['lr'])
        self.assertEqual(recorded['completed_epochs'],1)

    def _source_pair(self,root):
        for method in ('branch','davi'):
            run=root/'matched_time'/'caltech101'/method/'shots_16'/'seed_1'
            ck=run/'refactor_model'/'model.pth.tar-3'
            ck.parent.mkdir(parents=True)
            ck.write_bytes(f'{method} trained native state'.encode())
            h=sha256(ck)
            identity=dict(dataset='caltech101',method=method,shot=16,seed=1)
            manifest=dict(identity,study='matched_time',status='complete',native_max_epoch=50,
                batch_size=32,train_optimizer_name='adamw',train_lr=.001,
                train_lr_scheduler='cosine',train_warmup_epoch=1,
                train_precision='amp',cuda_device='RTX 5090',last_checkpoint_sha256=h)
            training=dict(budget_seconds=10.,elapsed_training_seconds=10.1,updates=10,
                budget_reached=True,last_step_seconds=0.2,last_checkpoint_epoch=3,
                last_checkpoint_sha256=h,completed_epochs=2,partial_epoch=True,
                last_update_batch_in_epoch=4)
            formal=dict(identity,partial=False,checkpoint=dict(selection='latest',epoch=3,checkpoint_sha256=h),
                n_mc_test=10,fusion_accuracy=92.,main_accuracy=91.,rep_accuracy=90.,
                rep_predictive_nll=.1,metrics=dict(ece=2.,aece=2.1))
            self._json(run/'manifest.json',manifest)
            self._json(run/'training.json',training)
            self._json(run/'formal/latest/epoch_3/complete/test.json',formal)
            for split,digest in [('train_fixed','identical-fixed-train'),('test','identical-test')]:
                r=dict(identity,partial=False,checkpoint=dict(selection='latest',epoch=3,checkpoint_sha256=h),
                    reference_schema=3,diagnostic_limit_batches=None,split=split,
                    s_ref=32,mc_repeats=2,n_images=20,input_sha256=digest,
                    objective_type='native_davi' if method=='davi' else 'native_avi_mc_only',
                    original_nll=.22,branch_nll=.29,gap1=.07,r_mean_fro=1.,r_trace=2.,t_trace=3.)
                self._json(run/f'diagnostics/latest/epoch_3/S32_R2/{split}/metrics.json',r)
        return root

    def _json(self,p,o):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(o),encoding='utf-8')

    def test_e5_matched_sources_train_test_and_partial_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._source_pair(Path(d));out=make_tables(root)
            self.assertEqual(out['e5_accepted_runs'],2)
            import csv
            with (root/'summary'/'table_s4_matched_runs.csv').open(newline='') as f:
                records=list(csv.DictReader(f))
            self.assertEqual({v['reference_split'] for v in records},{'train_fixed'})
            self.assertEqual({v['test_reference_input_sha256'] for v in records},{'identical-test'})
            self.assertEqual({v['partial_epoch'] for v in records},{'True'})
            self.assertEqual(len(records),2)

    def test_changed_checkpoint_rejected_without_false_match(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._source_pair(Path(d))
            ck=root/'matched_time/caltech101/davi/shots_16/seed_1/refactor_model/model.pth.tar-3'
            ck.write_bytes(b'new weights same epoch and seed')
            out=make_tables(root)
            self.assertEqual(out['e5_accepted_runs'],0)
            self.assertEqual(out['e5_rejected_pairs'],1)
            text=(root/'summary'/'table_s4_rejected_pairs.csv').read_text()
            self.assertIn('checkpoint SHA256',text)

    def test_missing_fixed_train_reference_rejected_not_promoted_test(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._source_pair(Path(d))
            ref=root/'matched_time/caltech101/davi/shots_16/seed_1/diagnostics/latest/epoch_3/S32_R2/train_fixed/metrics.json'
            ref.unlink()
            out=make_tables(root)
            self.assertEqual(out['e5_accepted_runs'],0)
            self.assertIn('missing train_fixed reference',
                (root/'summary'/'table_s4_rejected_pairs.csv').read_text())

    def test_changed_test_reference_digest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._source_pair(Path(d))
            ref=root/'matched_time/caltech101/davi/shots_16/seed_1/diagnostics/latest/epoch_3/S32_R2/test/metrics.json'
            obj=json.loads(ref.read_text());obj['input_sha256']='different-test';self._json(ref,obj)
            out=make_tables(root)
            self.assertEqual(out['e5_accepted_runs'],0)
            self.assertIn('test reference split/input digests differ',
                (root/'summary'/'table_s4_rejected_pairs.csv').read_text())

    def test_duplicate_sampling_reference_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._source_pair(Path(d))
            existing=root/'matched_time/caltech101/branch/shots_16/seed_1/diagnostics/latest/epoch_3/S32_R2'
            another=existing.parent/'S64_R2'
            import shutil
            shutil.copytree(existing,another)
            out=make_tables(root)
            self.assertEqual(out['e5_accepted_runs'],0)
            self.assertIn('multiple reference sample settings',
                (root/'summary'/'table_s4_rejected_pairs.csv').read_text())

if __name__=='__main__':unittest.main()
