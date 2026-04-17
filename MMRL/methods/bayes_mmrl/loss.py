from __future__ import annotations


class BayesMMRLLossAdapter:
    def __call__(self, outputs):
        return outputs.losses["total"]