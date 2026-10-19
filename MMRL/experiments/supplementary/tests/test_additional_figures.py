"""R13 three new mechanism figures: real-point provenance and no fake stages."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.supplementary.additional_figures import make_additional_figures
from experiments.supplementary.cli import budget_training_updates
from experiments.supplementary.tests.test_e5_budget import FakeTime,MockNativeTrainer


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj),encoding='utf-8')


def dummy_checkpoint(path, label):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(label.encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ThreeFigureTests(unittest.TestCase):
    def test_e2_gap_is_pure_branch_jensen_and_requires_multiple_real_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            run=root/'native/caltech101/davi/shots_16/seed_1'
            for ep,variance,gap in [(5,.31,.12),(25,.13,.04)]:
                cp=run/'trajectory/refactor_model'/f'model.pth.tar-{ep}'
                h=dummy_checkpoint(cp,str(ep))
                data=dict(dataset='caltech101',method='davi',shot=16,seed=1,
                    checkpoint=dict(epoch=ep,selection='trajectory',path=str(cp),checkpoint_sha256=h),
                    partial=False,reference_schema=3,split='train_fixed',input_sha256='stable',
                    s_ref=128,mc_repeats=2,diagnostic_batch_size=32,
                    main_sampled_var=variance/5,rep_sampled_var=variance,
                    main_empirical_jensen_gap=gap/10,rep_empirical_jensen_gap=gap,
                    # This large value must NOT influence the plot's pure Jensen gap.
                    gap2=800.)
                write(run/'diagnostics/trajectory'/f'epoch_{ep}'/'S128_R2/train_fixed/metrics.json',data)
            figs=[Path(x) for x in make_additional_figures(root) if 'S6_' in x]
            self.assertEqual(len(figs),1)
            self.assertTrue(figs[0].is_file())
            m=json.loads(figs[0].with_suffix('.json').read_text())
            self.assertEqual(len(m['checkpoints']),2)
            self.assertIn('pure_empirical',m['metric'])
            # Mutating a checkpoint invalidates it and leaves only one stage.
            (run/'trajectory/refactor_model/model.pth.tar-5').write_bytes(b'changed')
            self.assertEqual([x for x in make_additional_figures(root) if 'S6_' in x],[])

    def test_e3_noise_four_groups_and_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            run=root/'native/dtd/branch/shots_16/seed_2'
            cp=run/'refactor_model/model.pth.tar-50'
            h=dummy_checkpoint(cp,'branch-state')
            groups={k:dict(relative_variance=v,pairwise_cosine=c,mean_gradient_norm=.5,
                          valid_cosine_pairs=45,numel=10) for k,v,c in
                    [('phi',.5,.6),('r_mean',1.,.4),('r_rho',3.,0.),('t_rho',4.,-.1)]}
            data=dict(dataset='dtd',method='branch',shot=16,seed=2,partial=False,
                checkpoint=dict(path=str(cp),selection='latest',epoch=50,checkpoint_sha256=h),
                input={'input_sha256':'same'},noise_parameters={'draws':10,'batch_size':32,'noise_seed':7},
                groups=groups)
            write(run/'diagnostics/latest/epoch_50/gradient_noise/J10/gradient_noise.json',data)
            figures=[Path(x) for x in make_additional_figures(root) if 'S7_' in x]
            self.assertEqual(len(figures),1)
            self.assertEqual(json.loads(figures[0].with_suffix('.json').read_text())['parameter_groups'],
                             ['phi','r_mean','r_rho','t_rho'])
            cp.write_bytes(b'new')
            self.assertEqual([x for x in make_additional_figures(root) if 'S7_' in x],[])

    def test_e5_time_curve_pairs_exact_reference_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cps={}
            for method in ('branch','davi'):
                run=root/'matched_time/caltech101'/method/'shots_16/seed_1'
                write(run/'manifest.json',dict(dataset='caltech101',method=method,shot=16,seed=1,
                                               status='complete',study='matched_time'))
                snapshots=[]
                for phase in (0,25,50,75,100):
                    cp=run/'time_curve/refactor_model'/f'model.pth.tar-{phase}'
                    h=dummy_checkpoint(cp,f'{method}-{phase}')
                    snap=dict(phase=phase,train_wall_seconds=phase*2.,updates=phase*3,
                              real_epoch=phase//10,batch_in_epoch=1,checkpoint_sha256=h)
                    snapshots.append(snap);cps[(method,phase)]=cp
                    record=dict(dataset='caltech101',method=method,shot=16,seed=1,
                        checkpoint=dict(selection='time_curve',epoch=phase,path=str(cp),checkpoint_sha256=h),
                        split='train_fixed',partial=False,reference_schema=3,
                        n_images=20,s_ref=128,mc_repeats=1,input_sha256='fixed-both',
                        branch_nll=3.-phase/50. if method=='branch' else 2.8-phase/50.)
                    write(run/'diagnostics/time_curve'/f'epoch_{phase}'/'S128_R1/train_fixed/metrics.json',record)
                write(run/'training.json',dict(budget_seconds=200.,budget_reached=True,
                       time_curve_checkpoints=snapshots))
            figures=[Path(x) for x in make_additional_figures(root) if 'S8_' in x]
            self.assertEqual(len(figures),1)
            desc=json.loads(figures[0].with_suffix('.json').read_text())
            self.assertEqual(set(desc['sources']),{'branch','davi'})
            self.assertEqual(len(desc['sources']['davi']),5)
            self.assertEqual(desc['diagnostic_split'],'train_fixed')
            # Different fixed inputs cannot be graphed as matched objective.
            f=root/'matched_time/caltech101/davi/shots_16/seed_1/diagnostics/time_curve/epoch_25/S128_R1/train_fixed/metrics.json'
            m=json.loads(f.read_text());m['input_sha256']='different';write(f,m)
            self.assertEqual([x for x in make_additional_figures(root) if 'S8_' in x],[])

    def test_e5_snapshot_io_time_does_not_reduce_training_budget(self):
        t=FakeTime();tr=MockNativeTrainer(t,batches=15,epochs=1)
        events=[]
        def save(phase,elapsed,updates,epoch,batch):
            # Simulate a slow I/O callback; the training budget must exclude it.
            events.append((phase,updates,round(elapsed,4)))
            t.t+=100.
        result=budget_training_updates(tr,budget_seconds=10.,clock=t,snapshot_callback=save)
        self.assertEqual(result['updates'],10)
        self.assertEqual(result['time_curve_phases'],[0,25,50,75,100])
        self.assertEqual(result['excluded_snapshot_seconds'],400.)
        self.assertAlmostEqual(result['elapsed_training_seconds'],10.)
        self.assertEqual(result['overshoot_seconds'],0.)

if __name__=='__main__':unittest.main()

class NativeGradientNoiseSmokeTests(unittest.TestCase):
    """Actual differentiable Branch-like native posterior sample graph on CPU."""
    def test_four_native_parameter_groups_receive_draw_gradients(self):
        import torch
        import torch.nn.functional as F
        from torch import nn
        from types import SimpleNamespace
        from experiments.supplementary.tests.test_e3_gradients import _ToyModel
        from experiments.supplementary.diagnostics import mc_gradient_noise_by_group

        torch.manual_seed(812)
        model=_ToyModel()
        class TrainablePosterior(nn.Module):
            def __init__(self):
                super().__init__()
                self.posterior_mean=nn.Parameter(torch.eye(3)+torch.ones(3,3)*.02)
                self.posterior_rho=nn.Parameter(torch.ones(3)*-2.)
            def sample_many(self,n,use_mean=False):
                s=F.softplus(self.posterior_rho).view(1,1,3)
                return self.posterior_mean[None]+(0 if use_mean else s*torch.randn(n,3,3))
        model.image_encoder.bayes_proj_rep=TrainablePosterior()
        model.text_posterior=TrainablePosterior()
        def text_sample_features(eot,n,use_mean=False):
            w=model.text_posterior.sample_many(n,use_mean)
            samples=F.normalize(torch.einsum('cd,sdk->sck',eot,w),dim=-1)
            return samples,F.normalize(eot@model.text_posterior.posterior_mean,dim=-1)
        model.text_sample_features=text_sample_features
        def rep_samples(hidden,num_samples,use_mean=False):
            return torch.einsum('bd,sdk->sbk',hidden,model.image_encoder.bayes_proj_rep.sample_many(num_samples,use_mean))
        model.image_encoder.rep_samples=rep_samples
        trainer=SimpleNamespace(method=SimpleNamespace(model=model,device=torch.device('cpu'),
                   num_classes=3,cfg=SimpleNamespace(BAYESRT_MMRL=SimpleNamespace(ALPHA=.7))))
        batch={'img':torch.tensor([[.7,.2,.3],[.1,.8,.4],[.3,.2,.6]]),
               'label':torch.tensor([0,1,2])}
        result=mc_gradient_noise_by_group(trainer,batch,draws=5,seed=12345)
        self.assertEqual(set(result['groups']),{'phi','r_mean','r_rho','t_rho'})
        for g in result['groups'].values():
            self.assertEqual(g['total_possible_pairs'],10)
            self.assertEqual(g['valid_cosine_pairs'],10)
            self.assertGreater(g['numel'],0)
            self.assertIsNotNone(g['relative_variance'])
            self.assertIsNotNone(g['pairwise_cosine'])
            self.assertLessEqual(abs(g['pairwise_cosine']),1.00001)
        again=mc_gradient_noise_by_group(trainer,batch,draws=5,seed=12345)
        for key in result['groups']:
            self.assertAlmostEqual(result['groups'][key]['relative_variance'],
                                   again['groups'][key]['relative_variance'])
