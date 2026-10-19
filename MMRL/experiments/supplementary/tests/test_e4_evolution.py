"""R15 E4 evolution: exact provenance, same inputs and anti-stale pairing."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.supplementary.additional_figures import plot_e4_evolution
from experiments.supplementary.reporting import make_tables


class E4EvolutionTests(unittest.TestCase):
    def write(self, root, method, epoch, *, digest='same', draws=64):
        run=root/'native/caltech101'/method/'shots_16/seed_1'
        ckp=run/'trajectory/refactor_model'/f'model.pth.tar-{epoch}'
        ckp.parent.mkdir(parents=True,exist_ok=True)
        ckp.write_bytes((method+str(epoch)).encode())
        path=run/'diagnostics/trajectory'/f'epoch_{epoch}/e4_evolution'/f'images128_S{draws}_chunk16_B32_seed2027/metrics.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        m=dict(schema=1,method=method,dataset='caltech101',shot=16,seed=1,
               partial=False,rep_margin_type='mc_mean_logit',
               checkpoint=dict(selection='trajectory',epoch=epoch,path=str(ckp),checkpoint_sha256=hashlib.sha256(ckp.read_bytes()).hexdigest()),
               input=dict(input_sha256=digest),
               settings=dict(images=128,draws=draws,chunk=16,batch=32,input_seed=2027),
               r_mean_fro=float(epoch),r_trace=float(50-epoch/2),t_trace=float(.7-epoch/200),
               rep_margin_median=float(epoch)/10,rep_margin_mean=float(epoch)/8,
               rep_margin_positive_fraction=.5)
        path.write_text(json.dumps(m))
        return path,ckp

    def figures(self,root):
        out=root/'figures'; out.mkdir(exist_ok=True)
        saved=[]
        plot_e4_evolution(root,out,plt,np,saved)
        return saved

    def test_correct_three_method_pairing_and_source_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for method in ('original','branch','davi'):
                for ep in (5,25):self.write(root,method,ep)
            saved=self.figures(root)
            self.assertEqual(len(saved),1)
            pic=Path(saved[0]);self.assertTrue(pic.is_file())
            manifest=json.loads(pic.with_suffix('.json').read_text())
            self.assertEqual(manifest['stages'],[5,25])
            self.assertEqual(set(manifest['sources']),{'original','branch','davi'})
            self.assertEqual(manifest['rep_margin_type'],'mc_mean_logit')
            make_tables(root)
            with (root/'summary/e4_evolution_runs.csv').open() as f:
                self.assertEqual(len(f.readlines()),7)

    def test_stale_checkpoint_and_mismatched_input_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for method in ('original','branch','davi'):
                for ep in (5,25):self.write(root,method,ep)
            path,_=self.write(root,'branch',25,digest='DIFFERENT')
            self.assertEqual(len(self.figures(root)),0)
            self.write(root,'branch',25,digest='same')
            self.assertEqual(len(self.figures(root)),1)
            _,checkpoint=self.write(root,'branch',5)
            checkpoint.write_bytes(b'overwritten')
            self.assertEqual(len(self.figures(root)),0)
            make_tables(root)
            with (root/'summary/e4_evolution_runs.csv').open() as f:
                self.assertEqual(len(f.readlines()),6)

if __name__=='__main__':unittest.main()

class E4EvolutionMetricTests(unittest.TestCase):
    def test_fixed_input_mc_mean_logit_margin_and_cache(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import torch
        from experiments.supplementary import cli
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=Mock()
            method=SimpleNamespace(model=model, num_classes=2, device=torch.device('cpu'))
            trainer=SimpleNamespace(method=method)
            batch={'img':torch.zeros(3,3,4,4), 'label':torch.tensor([0,1,0])}
            input_info=dict(input_sha256='fixed',transform_protocol='native train_fixed deterministic test transform')
            args=SimpleNamespace(e4_images=3,e4_draws=5,e4_chunk=2,e4_batch=2,e4_input_seed=2027,
                                 run_dir=str(root),overwrite=False)
            ck=dict(selection='latest',epoch=50,checkpoint_sha256='native-ck')
            meta=dict(method='davi',dataset='caltech101',shot=16,seed=1,study='native')
            def sample(cache,k):
                size=cache['size']
                if size==2:
                    reps=torch.tensor([[6.,0.],[0.,4.]]).repeat(k,1,1)
                else:
                    reps=torch.tensor([[6.,0.]]).repeat(k,1,1)
                return torch.zeros_like(reps), reps
            def cached_enc(model,x,classes):
                return dict(size=x.shape[0])
            summary=dict(r_mean_fro=25.0,r_trace=17.,t_trace=.05)
            with patch.object(cli,'select_e3_fixed_batch',return_value=(batch,input_info)),\
                 patch('experiments.supplementary.core.cached_encode',side_effect=cached_enc),\
                 patch('experiments.supplementary.core.cached_draw',side_effect=sample) as sampler,\
                 patch('experiments.supplementary.diagnostics.posterior_summary',return_value=summary):
                cli._e4_evolution_eval(args,trainer,meta,ck)
                calls=sampler.call_count
                self.assertGreater(calls,0)
                cli._e4_evolution_eval(args,trainer,meta,ck)
                self.assertEqual(sampler.call_count,calls)
            p=root/'diagnostics/latest/epoch_50/e4_evolution/images3_S5_chunk2_B2_seed2027/metrics.json'
            record=json.loads(p.read_text())
            self.assertEqual(record['n_images'],3)
            self.assertEqual(record['rep_margin_median'],6.)
            self.assertEqual(record['rep_margin_positive_fraction'],1.)
            self.assertEqual(record['r_trace'],17.)
            self.assertEqual(record['input']['input_sha256'],'fixed')
