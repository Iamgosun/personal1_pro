from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from core.types import MethodOutputs


class BaseMethod(nn.Module):
    method_name = 'BaseMethod'

    def __init__(self, cfg, dm, device):
        super().__init__()
        self.cfg = cfg
        self.dm = dm
        self.device = device
        self.model = None
        self.loss = None

    def build(self):
        raise NotImplementedError

    def forward_train(self, batch) -> MethodOutputs:
        raise NotImplementedError

    def forward_eval(self, batch, eval_ctx) -> MethodOutputs:
        return self.forward_train(batch)

    def build_loss(self):
        return self.loss

    def get_optimizer_target(self):
        return self.model

    def supports_online(self) -> bool:
        return True

    def supports_cache(self) -> bool:
        return False

    def on_fit_start(self, trainer):
        return None

    def on_cache_ready(self, trainer):
        return None
