from __future__ import annotations


class BayesTextMMRLLossAdapter:
    def __call__(self, outputs):
        return outputs.losses["total"]