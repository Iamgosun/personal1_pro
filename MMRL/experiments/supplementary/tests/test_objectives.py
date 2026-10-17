import unittest
import torch
import torch.nn.functional as F
from experiments.supplementary.core import original_nll
from experiments.supplementary.diagnostics import sampled_ce, det_ce_per_image, _running_mean_and_m2


class Objectives(unittest.TestCase):
    def test_original_and_jensen(self):
        torch.manual_seed(123)
        c=torch.randn(13,7,9, dtype=torch.float64)
        r=torch.randn(13,7,9, dtype=torch.float64)
        y=torch.tensor([0,1,2,3,4,5,6])
        for alpha in (.2,.7,.95):
            original,branch,main,rep,rho=sampled_ce(c,r,y,alpha)
            self.assertTrue((branch-original>=-1.e-6).all())
            self.assertTrue((rho>=0).all() and (rho<=1).all())
            self.assertAlmostEqual(float(original.mean()),float(original_nll(c,r,y,alpha)),places=6)
            self.assertTrue(torch.allclose(branch,alpha*main+(1-alpha)*rep,atol=2e-7))

    def test_det_expression_matches_zero_variance_ce(self):
        torch.manual_seed(123)
        z=torch.randn(6,9)
        y=torch.tensor([1,2,3,4,5,8])
        zeros=torch.zeros_like(z)
        self.assertTrue(torch.allclose(det_ce_per_image(z,zeros,y),F.cross_entropy(z,y,reduction='none'),atol=1e-6))

    def test_davi_native_clamp_is_not_avi(self):
        mu=torch.zeros(2,3)
        var=torch.tensor([[0.,100.,0.],[0.,100.,0.]])
        y=torch.tensor([0,0])
        native=det_ce_per_image(mu,var,y,1.,50.)
        unclamped=det_ce_per_image(mu,var,y,1.,0.)
        self.assertTrue((unclamped-native>10.).all())

    def test_streaming_population_moments(self):
        torch.manual_seed(7)
        z=torch.randn(79,3,5,dtype=torch.float64)
        n=0;mu=m2=None
        for v in (z[:13],z[13:59],z[59:]):
            mu,m2=_running_mean_and_m2(n,mu,m2,v)
            n+=len(v)
        self.assertTrue(torch.allclose(mu,z.mean(0),rtol=1e-12,atol=1e-12))
        self.assertTrue(torch.allclose(m2/n,z.var(0,unbiased=False),rtol=1e-12,atol=1e-12))

if __name__=='__main__':unittest.main()
