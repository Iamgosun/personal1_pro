from typing import Iterable, List

VALID_TASKS = {"B2N", "FS", "CD"}
VALID_TRAINERS = {"MMRL", "MMRLpp"}


def resolve_task_from_args(opts: Iterable[str] | None, default: str = "B2N") -> str:
    if not opts:
        return default

    opts = list(opts)
    for idx, token in enumerate(opts[:-1]):
        if token == "TASK":
            return str(opts[idx + 1]).upper()
    return default


def _flatten_cfg(mapping: dict) -> List[str]:
    return [item for pair in mapping.items() for item in pair]


def get_dataset_specified_config(dataset: str, trainer: str, task: str) -> List[str]:
    task = str(task).upper()
    if task not in VALID_TASKS:
        raise AssertionError("The TASK must be either B2N, CD, or FS.")
    if trainer not in VALID_TRAINERS:
        # keep backward compatibility: silently ignore unknown trainers here
        return []

    if trainer == "MMRL":
        cfg = {
            "StanfordCars": {"TRAINER.MMRL.REG_WEIGHT": 7.0},
            "FGVCAircraft": {"TRAINER.MMRL.REG_WEIGHT": 6.0},
            "SUN397": {"TRAINER.MMRL.REG_WEIGHT": 6.0},
            "DescribableTextures": {"TRAINER.MMRL.REG_WEIGHT": 6.0},
            "Food101": {"TRAINER.MMRL.REG_WEIGHT": 5.0},
            "OxfordFlowers": {"TRAINER.MMRL.REG_WEIGHT": 4.0},
            "UCF101": {"TRAINER.MMRL.REG_WEIGHT": 3.0},
            "ImageNet": {"TRAINER.MMRL.REG_WEIGHT": 0.5},
            "Caltech101": {"TRAINER.MMRL.REG_WEIGHT": 0.5},
            "OxfordPets": {"TRAINER.MMRL.REG_WEIGHT": 0.2},
            "EuroSAT": {
                "TRAINER.MMRL.REP_DIM": 2048,
                "TRAINER.MMRL.REG_WEIGHT": 0.01,
            },
        }.get(dataset, {})
        return _flatten_cfg(cfg)

    if task in {"B2N", "FS"}:
        cfg = {
            "ImageNet": {"TRAINER.MMRLpp.BETA": 0.9, "TRAINER.MMRLpp.REG_WEIGHT": 0.2},
            "FGVCAircraft": {"TRAINER.MMRLpp.BETA": 0.9, "TRAINER.MMRLpp.REG_WEIGHT": 2.0},
            "UCF101": {"TRAINER.MMRLpp.BETA": 0.9, "TRAINER.MMRLpp.REG_WEIGHT": 3.0},
            "DescribableTextures": {"TRAINER.MMRLpp.BETA": 0.9, "TRAINER.MMRLpp.REG_WEIGHT": 7.0},
            "OxfordPets": {"TRAINER.MMRLpp.BETA": 0.7, "TRAINER.MMRLpp.REG_WEIGHT": 0.01},
            "StanfordCars": {"TRAINER.MMRLpp.BETA": 0.7, "TRAINER.MMRLpp.REG_WEIGHT": 6.0},
            "Caltech101": {"TRAINER.MMRLpp.BETA": 0.6, "TRAINER.MMRLpp.REG_WEIGHT": 3.0},
            "SUN397": {"TRAINER.MMRLpp.BETA": 0.5, "TRAINER.MMRLpp.REG_WEIGHT": 3.0},
            "OxfordFlowers": {"TRAINER.MMRLpp.BETA": 0.4, "TRAINER.MMRLpp.REG_WEIGHT": 7.0},
            "EuroSAT": {"TRAINER.MMRLpp.BETA": 0.2, "TRAINER.MMRLpp.REG_WEIGHT": 0.01},
            "Food101": {"TRAINER.MMRLpp.BETA": 0.1, "TRAINER.MMRLpp.REG_WEIGHT": 1.0},
        }.get(dataset, {})
        return _flatten_cfg(cfg)

    cfg = {
        "ImageNet": {"TRAINER.MMRLpp.BETA": 0.9, "TRAINER.MMRLpp.REG_WEIGHT": 0.1},
        "ImageNetV2": {"TRAINER.MMRLpp.BETA": 0.9},
        "ImageNetR": {"TRAINER.MMRLpp.BETA": 0.9},
        "ImageNetA": {"TRAINER.MMRLpp.BETA": 0.8},
        "ImageNetSketch": {"TRAINER.MMRLpp.BETA": 0.7},
        "FGVCAircraft": {"TRAINER.MMRLpp.BETA": 0.9},
        "UCF101": {"TRAINER.MMRLpp.BETA": 0.9},
        "SUN397": {"TRAINER.MMRLpp.BETA": 0.7},
        "OxfordPets": {"TRAINER.MMRLpp.BETA": 0.6},
        "Caltech101": {"TRAINER.MMRLpp.BETA": 0.6},
        "DescribableTextures": {"TRAINER.MMRLpp.BETA": 0.5},
        "OxfordFlowers": {"TRAINER.MMRLpp.BETA": 0.4},
        "StanfordCars": {"TRAINER.MMRLpp.BETA": 0.3},
        "EuroSAT": {"TRAINER.MMRLpp.BETA": 0.3},
        "Food101": {"TRAINER.MMRLpp.BETA": 0.3},
    }.get(dataset, {})
    return _flatten_cfg(cfg)
