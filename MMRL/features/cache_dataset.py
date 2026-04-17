from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, logits: Optional[torch.Tensor] = None):
        self.features = features
        self.labels = labels
        self.logits = logits

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        item = {'features': self.features[index], 'label': self.labels[index]}
        if self.logits is not None:
            item['logits'] = self.logits[index]
        return item
