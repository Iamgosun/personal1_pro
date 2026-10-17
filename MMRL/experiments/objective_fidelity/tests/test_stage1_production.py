"""When run inside full MMRL repository, compare Stage-1 math to production DAVI loss."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stage1_gap import surrogate


class ProductionParity(unittest.TestCase):
    def test_det_jensen_ce_parity(self):
        try:
            from methods.det_bayesrt_mmrl.losses import det_jensen_ce
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Requires complete MMRL methods tree: {e}")
        generator = torch.Generator().manual_seed(3399)
        mu = torch.randn(12, 17, generator=generator) * 2
        var = torch.rand(12, 17, generator=generator) * 140
        labels = torch.randint(0, 17, (12,), generator=generator)
        for eta, clip in ((1.0, 50.0), (0.5, 20.0), (1.0, 0.0), (0.0, 20.0)):
            got = surrogate(mu, var, labels, eta=eta, clamp=clip)
            ref = det_jensen_ce(mu, var, labels, var_scale=eta, var_clamp=clip)
            self.assertTrue(torch.allclose(got, ref.double(), atol=1e-4, rtol=5e-6),
                            f"eta={eta} clip={clip}: got={got}, production={ref}")


if __name__ == "__main__":
    unittest.main()
