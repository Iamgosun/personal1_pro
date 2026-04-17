from __future__ import annotations

from dassl.data.transforms import build_transform
from dassl.data.data_manager import DatasetWrapper


def build_split_dataset(cfg, data_source, is_train: bool):
    tfm = build_transform(cfg, is_train=is_train)
    return DatasetWrapper(cfg=cfg, data_source=data_source, transform=tfm, is_train=is_train)
