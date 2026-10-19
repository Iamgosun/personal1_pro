"""E3 selected-epoch restoration and comparison, no CLIP/GPU required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from experiments.supplementary.core import make_args, e3_checkpoint_request
from experiments.supplementary.reporting import make_tables
from experiments.supplementary.figures import make_figures

class E3SelectedEpochs(unittest.TestCase):
    def test_selected_snapshot_training_configuration_is_isolated(self):
        ns=make_args('caltech101','davi',16,1,'output','native','DATASETS',
                     e3_snapshot_epochs=(5,25))
        self.assertEqual(ns.trainer,'SupplementSelectedRunner')
        self.assertIn('TRAINER.NAME',ns.opts)
        self.assertNotIn('BAYESRT_MMRL.N_MC_TRAIN',ns.opts)
        self.assertNotIn('BAYESRT_MMRL.DET_VAR_CLAMP',ns.opts)
        with self.assertRaises(ValueError):
            make_args('caltech101','davi',16,1,'output','native','DATASETS',
                      trajectory=True,e3_snapshot_epochs=(5,25))

    def test_real_checkpoint_selection_and_missing_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); main=run/'refactor_model';main.mkdir()
            (main/'model.pth.tar-50').write_bytes(b'final')
            self.assertEqual(e3_checkpoint_request(run,'latest'),('latest',None))
            self.assertEqual(e3_checkpoint_request(run,'50'),('latest',None))
            with self.assertRaisesRegex(FileNotFoundError,'cannot reconstruct'):
                e3_checkpoint_request(run,'5')
            mid=run/'trajectory'/'refactor_model';mid.mkdir(parents=True)
            (mid/'model.pth.tar-5').write_bytes(b'early')
            (mid/'model.pth.tar-25').write_bytes(b'middle')
            self.assertEqual(e3_checkpoint_request(run,'5'),('trajectory',5))
            self.assertEqual(e3_checkpoint_request(run,'25'),('trajectory',25))
            with self.assertRaises(ValueError):e3_checkpoint_request(run,'0')

    def test_checkpoint_table_and_comparison_figure_from_distinct_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for epoch,sel,cos,err in ((5,'trajectory',.94,.21),
                                      (25,'trajectory',.97,.17),
                                      (50,'latest',.99,.13)):
                directory=(root/'native'/'caltech101'/'davi'/'shots_16'/'seed_1'/
                           'diagnostics'/sel/f'epoch_{epoch}'/'gradients'/
                           'refS512_chunk8_R5_B32_selstratified_seed2027_sizes1-5-10-50')
                directory.mkdir(parents=True)
                m=dict(dataset='caltech101',shot=16,seed=1,method='davi',
                    checkpoint=dict(selection=sel,epoch=epoch,checkpoint_sha256=str(epoch)*20),
                    reference_samples=512,reference_stable=True,
                    reference_repeat_quality=dict(cosine=.999,relative_error=.002),
                    davi=dict(cosine=cos,relative_error=err),
                    input=dict(batch_size_requested=32,n_images=32,n_unique_classes=32,
                               selection_strategy='stratified',selection_seed=2027,
                               input_sha256='a'*64),
                    e3_parameters=dict(reference_samples=512,chunk=8,repeats=5,
                                       batch_size=32,batch_strategy='stratified',input_seed=2027,
                                       mc_sizes=[1,5,10,50]),
                    curve=[dict(samples=s,repeat=r,cosine=1-.01/s+.00001*r,
                                relative_error=.06/(s**.5)+.0001*r)
                           for s in (1,5,10,50) for r in range(5)])
                (directory/'gradient.json').write_text(json.dumps(m),encoding='utf-8')
            stats=make_tables(root)
            self.assertEqual(stats['e3_checkpoint_rows'],12)
            with (root/'summary'/'e3_checkpoint_comparison.csv').open(newline='',encoding='utf-8') as f:
                rs=list(csv.DictReader(f))
            self.assertEqual({int(r['epoch']) for r in rs},{5,25,50})
            self.assertEqual(len(rs),12)
            imgs=[Path(p) for p in make_figures(root) if Path(p).name.startswith('S3_')]
            self.assertEqual(len(imgs),4)
            self.assertEqual(sum('S3_epoch_comparison_' in p.name for p in imgs),1)
            self.assertTrue(all(p.exists() for p in imgs))

class FullTrajectoryRunnerHook(unittest.TestCase):
    def test_initial_and_every_epoch_are_saved_without_skipping(self):
        import sys
        import types
        from unittest.mock import patch
        calls=[]
        class DummyRunner:
            def before_train(self): calls.append(('native-before',None))
            def after_epoch(self): calls.append(('native-after',self.epoch+1))
            def save_model(self,epoch,directory,is_best=False):
                calls.append(('save',epoch+1,Path(directory).name))
        class FakeRegistry:
            def __init__(self): self.items={}
            def registered_names(self): return list(self.items)
            def register(self):
                def wrapper(cls):self.items[cls.__name__]=cls;return cls
                return wrapper
            def get(self,name): return self.items[name]
        reg=FakeRegistry()
        fake_dassl=types.ModuleType('dassl')
        fake_engine=types.ModuleType('dassl.engine')
        fake_engine.TRAINER_REGISTRY=reg
        fake_trainers=types.ModuleType('trainers')
        fake_runner=types.ModuleType('trainers.refactor_runner')
        fake_runner.RefactorRunner=DummyRunner
        from experiments.supplementary.runner import register_trajectory
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules,{'dassl':fake_dassl,'dassl.engine':fake_engine,
                                         'trainers':fake_trainers,
                                         'trainers.refactor_runner':fake_runner}):
                register_trajectory()
                runner=reg.get('SupplementTrajectoryRunner')()
                runner.cfg=types.SimpleNamespace(OUTPUT_DIR=tmp)
                runner.before_train()
                for ep in range(1,6):
                    runner.epoch=ep-1
                    runner.after_epoch()
        self.assertEqual([x[1] for x in calls if x[0]=='save'],list(range(6)))
        self.assertEqual([x[1] for x in calls if x[0]=='native-after'],list(range(1,6)))
        self.assertTrue(all(x[2]=='trajectory' for x in calls if x[0]=='save'))

class E3SelectiveRunnerHook(unittest.TestCase):
    def test_saves_only_selected_epochs_and_initial_snapshot(self):
        import os
        import sys
        import types
        from unittest.mock import patch
        calls=[]
        class DummyRunner:
            def before_train(self):calls.append(('native-before',None))
            def after_epoch(self):calls.append(('native-after',self.epoch+1))
            def save_model(self,epoch,directory,is_best=False):
                calls.append(('save',epoch+1,Path(directory).name))
        class FakeRegistry:
            def __init__(self):self.items={}
            def registered_names(self):return list(self.items)
            def register(self):
                def deco(cls):self.items[cls.__name__]=cls;return cls
                return deco
            def get(self,name):return self.items[name]
        registry=FakeRegistry()
        fake_dassl=types.ModuleType('dassl')
        fake_engine=types.ModuleType('dassl.engine')
        fake_engine.TRAINER_REGISTRY=registry
        fake_trainers=types.ModuleType('trainers')
        fake_runner=types.ModuleType('trainers.refactor_runner')
        fake_runner.RefactorRunner=DummyRunner
        from experiments.supplementary.runner import register_trajectory
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules,{'dassl':fake_dassl,'dassl.engine':fake_engine,
                                         'trainers':fake_trainers,
                                         'trainers.refactor_runner':fake_runner}),\
                 patch.dict(os.environ,{'DAVI_SUPPLEMENT_E3_SAVE_EPOCHS':'5,25'}):
                register_trajectory()
                t=registry.get('SupplementSelectedRunner')()
                t.cfg=types.SimpleNamespace(OUTPUT_DIR=tmp)
                t.before_train()
                for ep in (1,4,5,6,25,26,50):
                    t.epoch=ep-1;t.after_epoch()
        self.assertEqual([v for v in calls if v[0]=='save'],
                         [('save',0,'trajectory'),('save',5,'trajectory'),('save',25,'trajectory')])
        self.assertEqual(len([v for v in calls if v[0]=='native-after']),7)

if __name__=='__main__':unittest.main()
