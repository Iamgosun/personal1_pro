from __future__ import annotations

from typing import List

VALID_TASKS = {'B2N', 'FS', 'CD'}
VALID_METHODS = {'MMRL', 'MMRLpp', 'MMRLPP'}


def _flatten_cfg(mapping: dict) -> List[str]:
    return [item for pair in mapping.items() for item in pair]


def resolve_method_dataset_overrides(method_name: str, dataset_name: str, protocol_name: str) -> List[str]:
    method_name = str(method_name)
    dataset_name = str(dataset_name)
    protocol_name = str(protocol_name).upper()

    if protocol_name not in VALID_TASKS:
        raise AssertionError('The PROTOCOL/TASK must be either B2N, CD, or FS.')
    if method_name not in VALID_METHODS:
        return []

    if method_name == 'MMRL':
        cfg = {
            'StanfordCars': {'MMRL.REG_WEIGHT': 7.0},
            'FGVCAircraft': {'MMRL.REG_WEIGHT': 6.0},
            'SUN397': {'MMRL.REG_WEIGHT': 6.0},
            'DescribableTextures': {'MMRL.REG_WEIGHT': 6.0},
            'Food101': {'MMRL.REG_WEIGHT': 5.0},
            'OxfordFlowers': {'MMRL.REG_WEIGHT': 4.0},
            'UCF101': {'MMRL.REG_WEIGHT': 3.0},
            'ImageNet': {'MMRL.REG_WEIGHT': 0.5},
            'Caltech101': {'MMRL.REG_WEIGHT': 0.5},
            'OxfordPets': {'MMRL.REG_WEIGHT': 0.2},
            'EuroSAT': {
                'MMRL.REP_DIM': 2048,
                'MMRL.REG_WEIGHT': 0.01,
            },
        }.get(dataset_name, {})
        return _flatten_cfg(cfg)

    if protocol_name in {'B2N', 'FS'}:
        cfg = {
            'ImageNet': {'MMRLPP.BETA': 0.9, 'MMRLPP.REG_WEIGHT': 0.2},
            'FGVCAircraft': {'MMRLPP.BETA': 0.9, 'MMRLPP.REG_WEIGHT': 2.0},
            'UCF101': {'MMRLPP.BETA': 0.9, 'MMRLPP.REG_WEIGHT': 3.0},
            'DescribableTextures': {'MMRLPP.BETA': 0.9, 'MMRLPP.REG_WEIGHT': 7.0},
            'OxfordPets': {'MMRLPP.BETA': 0.7, 'MMRLPP.REG_WEIGHT': 0.01},
            'StanfordCars': {'MMRLPP.BETA': 0.7, 'MMRLPP.REG_WEIGHT': 6.0},
            'Caltech101': {'MMRLPP.BETA': 0.6, 'MMRLPP.REG_WEIGHT': 3.0},
            'SUN397': {'MMRLPP.BETA': 0.5, 'MMRLPP.REG_WEIGHT': 3.0},
            'OxfordFlowers': {'MMRLPP.BETA': 0.4, 'MMRLPP.REG_WEIGHT': 7.0},
            'EuroSAT': {'MMRLPP.BETA': 0.2, 'MMRLPP.REG_WEIGHT': 0.01},
            'Food101': {'MMRLPP.BETA': 0.1, 'MMRLPP.REG_WEIGHT': 1.0},
        }.get(dataset_name, {})
        return _flatten_cfg(cfg)

    cfg = {
        'ImageNet': {'MMRLPP.BETA': 0.9, 'MMRLPP.REG_WEIGHT': 0.1},
        'ImageNetV2': {'MMRLPP.BETA': 0.9},
        'ImageNetR': {'MMRLPP.BETA': 0.9},
        'ImageNetA': {'MMRLPP.BETA': 0.8},
        'ImageNetSketch': {'MMRLPP.BETA': 0.7},
        'FGVCAircraft': {'MMRLPP.BETA': 0.9},
        'UCF101': {'MMRLPP.BETA': 0.9},
        'SUN397': {'MMRLPP.BETA': 0.7},
        'OxfordPets': {'MMRLPP.BETA': 0.6},
        'Caltech101': {'MMRLPP.BETA': 0.6},
        'DescribableTextures': {'MMRLPP.BETA': 0.5},
        'OxfordFlowers': {'MMRLPP.BETA': 0.4},
        'StanfordCars': {'MMRLPP.BETA': 0.3},
        'EuroSAT': {'MMRLPP.BETA': 0.3},
        'Food101': {'MMRLPP.BETA': 0.3},
    }.get(dataset_name, {})
    return _flatten_cfg(cfg)
