"""CPU-only tests; run from MMRL as: python -m unittest discover -s experiments/objective_fidelity/tests -p 'test_stage1_*.py'"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from stage1_gap import decompose_batch, surrogate, weighted_aggregate


class TestStage1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(301)

    @staticmethod
    def evaluate(main, rep, labels, alpha=0.7, eta=1., clamp=50., mu_m=None, var_m=None, mu_r=None, var_r=None, chunks=4):
        def parts():
            for start in range(0, len(main), chunks):
                yield main[start:start+chunks], rep[start:start+chunks]
        if mu_m is None:
            mu_m = main.mean(0)
        if var_m is None:
            var_m = main.var(0, unbiased=False)
        if mu_r is None:
            mu_r = rep.mean(0)
        if var_r is None:
            var_r = rep.var(0, unbiased=False)
        return decompose_batch(parts(), labels, mu_m, var_m, mu_r, var_r, alpha, eta, clamp)

    def test_six_term_identity_and_two_jensens(self):
        main = 4*torch.randn(51,4,8)
        rep = 2*torch.randn(51,4,8) + 1.
        labels = torch.tensor([0,4,2,6])
        d = self.evaluate(main,rep,labels,eta=0.7,clamp=1.2,chunks=7)
        self.assertGreaterEqual(d['mix_gap'], -1e-9)
        self.assertGreaterEqual(d['softmax_jensen_gap'], -1e-9)
        self.assertLessEqual(d['clamp_error'], 1e-9)
        self.assertAlmostEqual(d['total_gap'], sum(d[k] for k in (
            'mix_gap','softmax_jensen_gap','gaussian_error',
            'propagation_error','scale_error','clamp_error')),places=9)
        self.assertEqual(d['mc_samples'],51)

    def test_identical_branches_mixture_gap_zero(self):
        z = torch.randn(19,3,5)
        d = self.evaluate(z,z,torch.tensor([1,2,3]))
        self.assertAlmostEqual(d['mix_gap'],0,places=10)

    def test_constant_logits_zero_jensen_gap(self):
        z = torch.tensor([[[2.,0.],[1.,2.]]]).expand(23,-1,-1).contiguous()
        y = torch.tensor([0,1])
        d = self.evaluate(z,z,y)
        self.assertAlmostEqual(d['softmax_jensen_gap'],0,places=10)
        self.assertAlmostEqual(d['total_gap'],0,places=10)

    def test_eta_separated_from_moment_propagation(self):
        m = torch.randn(50,3,7)
        r = torch.randn(50,3,7)
        y = torch.tensor([0,3,6])
        d = self.evaluate(m,r,y,eta=0.,clamp=0.)
        self.assertLessEqual(d['scale_error'],1e-9)
        self.assertAlmostEqual(d['clamp_error'],0,places=10)
        d2 = self.evaluate(m,r,y,eta=1.,clamp=0.)
        self.assertAlmostEqual(d2['scale_error'],0,places=10)

    def test_chunk_invariance(self):
        main = torch.randn(65,2,4)*8
        rep = torch.randn(65,2,4)*5
        y=torch.tensor([1,3])
        a=self.evaluate(main,rep,y,chunks=1)
        b=self.evaluate(main,rep,y,chunks=33)
        for key in ('original_nll','branch_nll','empirical_jensen_nll','gaussian_moment_nll',
                    'mix_gap','softmax_jensen_gap','gaussian_error','total_gap'):
            self.assertAlmostEqual(a[key],b[key],places=9)

    def test_weighted_aggregate(self):
        z1=torch.randn(16,2,4)
        z2=torch.randn(16,3,4)
        a=self.evaluate(z1,z1+0.5,torch.tensor([0,1]))
        b=self.evaluate(z2,z2+0.5,torch.tensor([0,1,2]))
        total=weighted_aggregate([(2,a),(3,b)])
        self.assertAlmostEqual(total['original_nll'],(2*a['original_nll']+3*b['original_nll'])/5)
        self.assertEqual(total['num_examples'],5)

    def test_mixture_gap_is_kl_of_branch_responsibilities(self):
        alpha=.7
        logits_m=torch.randn(100,4,6)
        logits_r=torch.randn(100,4,6)
        idx=torch.tensor([0,2,1,4])[None,:,None].expand(100,-1,1)
        p_m=torch.softmax(logits_m,dim=-1).gather(-1,idx).squeeze(-1)
        p_r=torch.softmax(logits_r,dim=-1).gather(-1,idx).squeeze(-1)
        mix=alpha*p_m+(1-alpha)*p_r
        responsibility=alpha*p_m/mix
        kl=alpha*torch.log(alpha/responsibility)+(1-alpha)*torch.log((1-alpha)/(1-responsibility))
        direct=torch.log(mix)-alpha*torch.log(p_m)-(1-alpha)*torch.log(p_r)
        self.assertTrue(torch.allclose(kl,direct,atol=1e-5,rtol=1e-5))

    def test_scale_clip_surrogate(self):
        mu=torch.randn(5,7)
        var=torch.abs(torch.randn(5,7))*100
        y=torch.tensor([0,2,3,4,6])
        actual=surrogate(mu,var,y,eta=1.,clamp=50.)
        formula=(-mu.double().gather(1,y[:,None]).mean()
                 +torch.logsumexp(mu.double()+0.5*var.double().clamp(max=50.),dim=1).mean())
        self.assertAlmostEqual(float(actual),float(formula),places=10)


if __name__=='__main__':
    unittest.main()
