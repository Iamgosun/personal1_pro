"""Check the three paper-level objectives, not just detailed error bookkeeping."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_gap import decompose_batch, weighted_aggregate


class ObjectiveFidelity(unittest.TestCase):
    def test_original_mixture_branch_and_davi_formula(self):
        torch.manual_seed(42)
        m = torch.randn(32, 3, 6) * 3
        r = torch.randn(32, 3, 6) * 2 + 0.25
        y = torch.tensor([0, 2, 4])
        alpha = .7
        mm, vm = m.mean(0), m.var(0, unbiased=False)
        mr, vr = r.mean(0), r.var(0, unbiased=False)
        result = decompose_batch([(m[:11], r[:11]), (m[11:], r[11:])],
                                 y, mm, vm, mr, vr, alpha, eta=1., clamp=50.)
        im = torch.arange(len(y))
        logm = F.log_softmax(m.double(), -1)[:, im, y]
        logr = F.log_softmax(r.double(), -1)[:, im, y]
        o = -torch.logaddexp(logm + torch.tensor(alpha).log(),
                             logr + torch.tensor(1-alpha).log()).mean()
        b = -(alpha * logm.mean() + (1 - alpha) * logr.mean())
        def det(mu, v):
            return (torch.logsumexp(mu.double() + .5*v.double().clamp_max(50), -1)
                    - mu.double()[im, y]).mean()
        d = alpha*det(mm, vm) + (1-alpha)*det(mr, vr)
        self.assertAlmostEqual(result['original_nll'], float(o), places=6)
        self.assertAlmostEqual(result['branch_nll'], float(b), places=6)
        self.assertAlmostEqual(result['davi_nll'], float(d), places=6)
        aggregate = weighted_aggregate([(len(y), result)])
        self.assertAlmostEqual(aggregate['det_gap'], float(d-b), places=6)
        self.assertGreaterEqual(aggregate['mae_det_gap'] + 1e-9, abs(aggregate['det_gap']))


if __name__ == '__main__':
    unittest.main()
