"""Regression for E1: no native model-best despite complete epoch snapshots."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from experiments.supplementary.core import checkpoint_path
from experiments.supplementary.reporting import make_tables


class NativeE1Checkpoint(unittest.TestCase):
    def test_last_native_checkpoint_does_not_require_model_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            native = run/'refactor_model'
            native.mkdir()
            (native/'model.pth.tar-12').write_bytes(b'last')
            (native/'model.pth.tar-2').write_bytes(b'earlier')
            root, checkpoint, epoch = checkpoint_path(run, 'latest')
            self.assertEqual(root, run)
            self.assertEqual(checkpoint.name, 'model.pth.tar-12')
            self.assertEqual(epoch, 12)
            with self.assertRaisesRegex(FileNotFoundError, 'model-best'):
                checkpoint_path(run, 'best')

    def test_trace_snapshot_does_not_depend_on_native_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            trajectory = run/'trajectory'/'refactor_model'
            trajectory.mkdir(parents=True)
            (trajectory/'model.pth.tar-0').write_bytes(b'initial')
            (trajectory/'model.pth.tar-1').write_bytes(b'epoch1')
            root, checkpoint, epoch = checkpoint_path(run, 'trajectory', 1)
            self.assertEqual(root, run/'trajectory')
            self.assertEqual(checkpoint.name, 'model.pth.tar-1')
            self.assertEqual(epoch, 1)

    def test_sparse_partial_epochs_allowed(self):
        from experiments.supplementary.core import available_trajectory_epochs, e1_trace_is_current
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = run/'trajectory'/'refactor_model'
            path.mkdir(parents=True)
            for epoch in (0, 2, 5):
                (path/f'model.pth.tar-{epoch}').write_bytes(b'checkpoint')
            self.assertEqual(available_trajectory_epochs(run), [0, 2, 5])
            root, checkpoint, epoch = checkpoint_path(run, 'trajectory', 0)
            self.assertEqual(checkpoint.name, 'model.pth.tar-0')
            self.assertEqual(epoch, 0)
            self.assertFalse(e1_trace_is_current(run,10,4,32))
            trace=run/'trace';trace.mkdir()
            (trace/'trajectory.csv').write_text('epoch\n0\n2\n5\n',encoding='utf-8')
            (trace/'trajectory.json').write_text(json.dumps({
                'e1_samples':10,'e1_batches':4,'batch_size':32,
                'epochs_evaluated':[0,2,5]}),encoding='utf-8')
            self.assertTrue(e1_trace_is_current(run,10,4,32))
            self.assertFalse(e1_trace_is_current(run,50,4,32))
            self.assertFalse(e1_trace_is_current(run,10,0,32))
            self.assertFalse(e1_trace_is_current(run,10,4,64))
            (path/'model.pth.tar-3').write_bytes(b'another checkpoint')
            self.assertFalse(e1_trace_is_current(run,10,4,32))

    def test_only_initial_snapshot_can_be_used_as_pilot(self):
        from experiments.supplementary.core import available_trajectory_epochs
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            traj=run/'trajectory'/'refactor_model';traj.mkdir(parents=True)
            (traj/'model.pth.tar-0').write_bytes(b'init')
            self.assertEqual(available_trajectory_epochs(run),[0])
            self.assertEqual(checkpoint_path(run,'trajectory',0)[2],0)

    def test_s1_summary_accepts_truthfully_labeled_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method in ('original', 'branch', 'davi'):
                d = root/'native'/'caltech101'/method/'shots_16'/'seed_1'/'formal'/'latest'/'epoch_50'/'complete'
                d.mkdir(parents=True)
                value = dict(dataset='caltech101',method=method,seed=1,shot=16,
                             partial=False,checkpoint=dict(selection='latest',epoch=50),
                             n_mc_test=10,main_accuracy=80.,rep_accuracy=78.,
                             fusion_accuracy=81.,rep_predictive_nll=.3,
                             metrics={'ece':2.3,'aece':2.2})
                (d/'test.json').write_text(json.dumps(value),encoding='utf-8')
            summary = make_tables(root)
            self.assertEqual(summary['s1_runs'], 3)
            with (root/'summary'/'table_s1_runs.csv').open(encoding='utf-8',newline='') as f:
                results = list(csv.DictReader(f))
            self.assertEqual({r['method'] for r in results},{'original','branch','davi'})
            self.assertEqual({r['checkpoint'] for r in results},{'latest_epoch_50'})


if __name__=='__main__':
    unittest.main()
