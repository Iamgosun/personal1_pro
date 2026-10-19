"""E4 source-faithful metrics, cache identity and strict three-method pairing."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from experiments.supplementary.diagnostics import mc_mean_logit_margin
from experiments.supplementary.reference_cache import (
    REFERENCE_SCHEMA, initial_snapshot_hash, is_current, formal_is_current)
from experiments.supplementary.figures import make_figures
from experiments.supplementary.reporting import make_tables


class E4MechanismTests(unittest.TestCase):
    def test_rep_margin_is_of_mc_mean_logits_not_bma_logp(self):
        # Sampled logits: the chosen category has marginally larger mean logit
        # yet is downweighted in one highly confident posterior sample.
        logits=torch.tensor([[[12., 0.]], [[0., 10.]]])
        mean_logits=logits.double().mean(0)
        labels=torch.tensor([0])
        margin=mc_mean_logit_margin(mean_logits,labels)
        self.assertAlmostEqual(margin.item(),1.)
        probs=torch.softmax(logits,-1).double().mean(0)
        bma_logp_margin=(probs[:,0]/probs[:,1]).log()
        self.assertNotAlmostEqual(margin.item(),bma_logp_margin.item(),places=2)
        with self.assertRaises(ValueError):
            mc_mean_logit_margin(torch.tensor([[1.]]),labels)

    def test_cache_requires_current_checkpoint_initialization_and_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            init=root/'trajectory/refactor_model/model.pth.tar-0'
            init.parent.mkdir(parents=True);init.write_bytes(b'initial')
            ck=root/'refactor_model/model.pth.tar-50'
            ck.parent.mkdir(parents=True);ck.write_bytes(b'original')
            digest=lambda x:hashlib.sha256(x).hexdigest()
            sub=root/'diagnostics/latest/epoch_50/S1024_R2/test';sub.mkdir(parents=True)
            payload=dict(reference_schema=REFERENCE_SCHEMA,split='test',partial=False,
                checkpoint={'checkpoint_sha256':digest(b'original')},
                initial_checkpoint_sha256=initial_snapshot_hash(root),
                s_ref=1024,mc_repeats=2,diagnostic_batch_size=32,
                diagnostic_limit_batches=None,mc_reference_tolerance=0.03,
                rep_margin_type='mc_mean_logit',
                rep_logp_margin_type='posterior_predictive_log_probability')
            (sub/'metrics.json').write_text(json.dumps(payload))
            (sub/'per_image.csv').write_text('rep_margin,visual_clip_cos\n1,.5\n')
            (sub/'independent_mc_repeats.json').write_text('[]')
            options=dict(checkpoint_sha256=digest(b'original'),
                initial_sha256=initial_snapshot_hash(root),samples=1024,repeats=2,
                batch_size=32,tolerance=.03,split='test',require_e4=True)
            self.assertTrue(is_current(sub/'metrics.json',**options))
            self.assertFalse(is_current(sub/'metrics.json',**(options|{'batch_size':64})))
            self.assertFalse(is_current(sub/'metrics.json',**(options|{'samples':512})))
            self.assertFalse(is_current(sub/'metrics.json',**(options|{'checkpoint_sha256':digest(b'new')})))
            self.assertFalse(is_current(sub/'metrics.json',**(options|{'initial_sha256':digest(b'changed-init')})))
            self.assertFalse(is_current(sub/'metrics.json',**(options|{'require_vectors':True})))
            (sub/'moment_vectors.json').write_text('[]')
            self.assertTrue(is_current(sub/'metrics.json',**(options|{'require_vectors':True})))
            payload['rep_margin_type']='posterior_predictive_log_probability'
            (sub/'metrics.json').write_text(json.dumps(payload))
            self.assertFalse(is_current(sub/'metrics.json',**options))
            payload['rep_margin_type']='mc_mean_logit';payload.pop('reference_schema')
            (sub/'metrics.json').write_text(json.dumps(payload))
            self.assertFalse(is_current(sub/'metrics.json',**options))

    def test_formal_result_requires_matching_weight_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'test.json'
            path.write_text(json.dumps({'partial':False,'checkpoint':{'checkpoint_sha256':'a'}}))
            self.assertTrue(formal_is_current(path,'a'))
            self.assertFalse(formal_is_current(path,'b'))
            path.write_text(json.dumps({'partial':True,'checkpoint':{'checkpoint_sha256':'a'}}))
            self.assertFalse(formal_is_current(path,'a'))

    def _make_run(self,root,method,epoch,*,input_hash='fixed',samples=32,repeats=2):
        run=root/f'native/caltech101/{method}/shots_16/seed_1'
        ck=run/'refactor_model'/f'model.pth.tar-{epoch}'
        ck.parent.mkdir(parents=True,exist_ok=True)
        ck.write_bytes((method+str(epoch)).encode())
        src_hash=hashlib.sha256(ck.read_bytes()).hexdigest()
        sub=run/'diagnostics/latest'/f'epoch_{epoch}'/f'S{samples}_R{repeats}'/'test'
        sub.mkdir(parents=True)
        info=dict(dataset='caltech101',shot=16,seed=1,method=method,partial=False,
            reference_schema=REFERENCE_SCHEMA,rep_margin_type='mc_mean_logit',
            split='test',s_ref=samples,mc_repeats=repeats,input_sha256=input_hash,
            checkpoint={'selection':'latest','epoch':epoch,'path':str(ck),'checkpoint_sha256':src_hash},
            initial_checkpoint_sha256=None,text_clip_cos_per_class=[.91,.97],
            r_mean_fro=5.,r_trace=2.,t_trace=.3,visual_clip_cos=.8,text_clip_cos_mean=.94,
            rep_margin_median=.5,original_nll=.2,branch_nll=.3,gap1=.1,
            objective_type='native_avi_mc_only' if method!='davi' else 'native_davi')
        (sub/'metrics.json').write_text(json.dumps(info))
        with (sub/'per_image.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['visual_clip_cos','rep_margin'])
            w.writeheader();w.writerows([dict(visual_clip_cos=.8,rep_margin=1),
                                         dict(visual_clip_cos=.5,rep_margin=-.1)])
        return ck,sub

    def test_s4_strict_epoch_input_sampling_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for epoch in (5,25):
                for method in ('original','branch','davi'):
                    self._make_run(root,method,epoch)
            figures=[Path(x) for x in make_figures(root) if '/S4_' in x]
            self.assertEqual(len(figures),2,figures)
            self.assertTrue(any('epoch5_' in str(p) for p in figures))
            self.assertTrue(any('epoch25_' in str(p) for p in figures))
            for png in figures:
                self.assertGreater(png.stat().st_size,1000)
                manifest=json.loads(png.with_suffix('.json').read_text())
                self.assertEqual(set(manifest['source_by_method']),{'original','branch','davi'})
                self.assertEqual(manifest['rep_margin_type'],'mc_mean_logit')
            # Mismatched test inputs cannot be silently paired.
            mpath=root/'native/caltech101/branch/shots_16/seed_1/diagnostics/latest/epoch_25/S32_R2/test/metrics.json'
            info=json.loads(mpath.read_text());info['input_sha256']='different';mpath.write_text(json.dumps(info))
            now=[x for x in make_figures(root) if '/S4_' in x]
            self.assertEqual(len(now),1)
            # The old PNG remains on disk but isn't returned as a current figure.
            tables=make_tables(root)
            self.assertEqual(tables['s3_runs'],6)
            with (root/'summary/table_s3_runs.csv').open(newline='') as f:
                records=list(csv.DictReader(f))
            self.assertEqual(len(records),6)
            # Same-seed retrain: changed checkpoint bytes without recomputing
            # JSON invalidates the run for BOTH S4 and Table S3.
            ck=root/'native/caltech101/branch/shots_16/seed_1/refactor_model/model.pth.tar-5'
            ck.write_bytes(b'retrained')
            new=[x for x in make_figures(root) if '/S4_' in x]
            self.assertEqual(len(new),0)
            make_tables(root)
            with (root/'summary/table_s3_runs.csv').open(newline='') as f:
                records=list(csv.DictReader(f))
            self.assertEqual(len(records),5)

    def test_s4_skips_legacy_bma_only_result(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for method in ('original','branch','davi'):
                _,sub=self._make_run(root,method,50)
                if method=='branch':
                    data=json.loads((sub/'metrics.json').read_text())
                    data['rep_margin_type']='posterior_predictive_log_probability'
                    (sub/'metrics.json').write_text(json.dumps(data))
            self.assertEqual([x for x in make_figures(root) if '/S4_' in x],[])
            make_tables(root)
            with (root/'summary/table_s3_runs.csv').open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))),2)

if __name__=='__main__':unittest.main()
