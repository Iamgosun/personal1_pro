"""Pure mathematical E6 checks; run from MMRL root: python -m unittest discover -s experiments/supplementary/tests -p 'test_e6.py'"""
import unittest
import torch
import torch.nn.functional as F
from experiments.supplementary import e6_analysis as a


class TestE6Math(unittest.TestCase):
    def test_identical_features(self):
        h=torch.eye(3,4)
        w=torch.eye(4,3)
        f=a.cross_project(h,h,w,w)
        self.assertAlmostEqual(a.row_cos(f['AA'],f['DD']),1,places=6)
        self.assertIsNone(a.directional_attribution(f)['head_pct'])

    def test_only_head_change(self):
        h=torch.eye(3)
        m_a=torch.eye(3)
        m_d=-torch.eye(3)
        f=a.cross_project(h,h,m_a,m_d)
        out=a.directional_attribution(f)
        self.assertAlmostEqual(out['head_pct'],100,places=7)
        self.assertAlmostEqual(out['hidden_pct'],0,places=7)

    def test_only_hidden_change(self):
        h_a=torch.eye(3)
        h_d=-torch.eye(3)
        m=torch.eye(3)
        out=a.directional_attribution(a.cross_project(h_a,h_d,m,m))
        self.assertAlmostEqual(out['head_pct'],0,places=7)
        self.assertAlmostEqual(out['hidden_pct'],100,places=7)

    def test_interaction_identity(self):
        g=torch.Generator().manual_seed(115)
        ha=torch.randn((6,4),generator=g)
        hd=torch.randn((6,4),generator=g)
        ma=torch.randn((4,3),generator=g)
        md=torch.randn((4,3),generator=g)
        out=a.directional_attribution(a.cross_project(ha,hd,ma,md))
        self.assertAlmostEqual(out['head_pct']+out['hidden_pct'],100,places=6)

    def test_zero_projection_is_undefined(self):
        h=torch.eye(3)
        self.assertIsNone(a.normalize_project(h,torch.zeros(3,3)))
        out=a.directional_attribution(a.cross_project(h,h,torch.zeros(3,3),torch.eye(3)))
        self.assertIsNone(out['head_pct'])

    def test_deterministic_gradients_not_mc_variance(self):
        stats=a.draw_statistics([torch.tensor([1.,2.])])
        self.assertIsNone(stats['relative_variance'])
        self.assertIsNone(stats['pairwise_cos'])
        self.assertIsNone(stats['cancellation_retention'])

    def test_canceling_draw_gradients(self):
        stats=a.draw_statistics([torch.tensor([1.,0.]),torch.tensor([-1.,0.])])
        self.assertAlmostEqual(stats['cancellation_retention'],0)
        self.assertAlmostEqual(stats['pairwise_cos'],-1)
        self.assertIsNone(stats['relative_variance'])

    def test_identical_draws(self):
        stats=a.draw_statistics([torch.tensor([1.,1.]),torch.tensor([1.,1.])])
        self.assertAlmostEqual(stats['cancellation_retention'],1)
        self.assertAlmostEqual(stats['relative_variance'],0)

    def test_calibration_perfect_confidence(self):
        probs=torch.tensor([[.9,.1],[.1,.9]])
        labels=torch.tensor([0,1])
        stats=a.calibration_metrics(probs,labels)
        self.assertAlmostEqual(stats['acc'],1)
        self.assertAlmostEqual(stats['ece_15_equal_width'],.1,places=6)
        self.assertAlmostEqual(stats['aece_15_equal_count'],.1,places=6)

    def test_sigma_inverse(self):
        sig=torch.tensor([[.01,.0005]])
        for gamma in [0.0,.5,1.,2.]:
            rho=a._rho_for_scaled_sigma(sig,1e-6,gamma)
            back=F.softplus(rho)+1e-6
            if gamma:
                self.assertTrue(torch.allclose(back,sig*gamma,atol=1e-7))
            else:
                self.assertTrue(torch.all(back<=1.01e-6))

if __name__=='__main__':unittest.main()
