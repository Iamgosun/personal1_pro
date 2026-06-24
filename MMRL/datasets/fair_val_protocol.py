from __future__ import annotations

import os
from typing import Any, Iterable, Optional


_PATCH_INSTALLED = False
_ORIGINAL_DATASETBASE_INIT = None


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _enabled_from_method_config() -> bool:
    return _truthy(os.environ.get("MMRL_MERGE_VAL_TO_TRAIN"), False)


def _drop_val_from_method_config() -> bool:
    return _truthy(os.environ.get("MMRL_DROP_VAL_AFTER_MERGE"), True)


def _item_key(item: Any):
    return (
        getattr(item, "impath", None),
        getattr(item, "label", None),
        getattr(item, "classname", None),
    )


def _looks_same_dataset(a: Optional[Iterable], b: Optional[Iterable]) -> bool:
    """Conservative check to avoid merging test-as-val into train."""
    if a is None or b is None:
        return False
    if a is b:
        return True

    try:
        la = list(a)
        lb = list(b)
    except TypeError:
        return False

    if len(la) == 0 or len(lb) == 0 or len(la) != len(lb):
        return False

    idxs = [0, len(la) // 2, len(la) - 1]
    return all(_item_key(la[i]) == _item_key(lb[i]) for i in idxs)


def _safe_to_merge(train_x, val, test) -> bool:
    if val is None:
        print("[KPlusValProtocol] SKIP merge: val is None.")
        return False

    try:
        if len(val) == 0:
            print("[KPlusValProtocol] SKIP merge: val is empty.")
            return False
    except TypeError:
        print("[KPlusValProtocol] SKIP merge: val has no length.")
        return False

    if _looks_same_dataset(val, test):
        print(
            "[KPlusValProtocol] SKIP merge: val appears identical to test; "
            "merging would leak evaluation samples into training."
        )
        return False

    if train_x is None:
        print("[KPlusValProtocol] SKIP merge: train_x is None.")
        return False

    return True


def install_kplusval_datasetbase_patch(cfg: Any) -> None:
    """
    Method-config-driven CLAP Tab.5 fair K+Val-shot protocol.

    The method YAML may contain:

        DATASET:
          MERGE_VAL_TO_TRAIN: true
          DROP_VAL_AFTER_MERGE: true

    run.py reads these flags before YACS merges the YAML, removes them from a
    sanitized temporary method YAML, and exports them as runtime env variables.
    This keeps run_plan.sh protocol-agnostic and avoids core/config.py edits.
    """
    global _PATCH_INSTALLED, _ORIGINAL_DATASETBASE_INIT

    if not _enabled_from_method_config():
        return

    if _PATCH_INSTALLED:
        return

    from dassl.data.datasets import DatasetBase

    _ORIGINAL_DATASETBASE_INIT = DatasetBase.__init__

    def patched_init(self, *args, **kwargs):
        train_x = kwargs.get("train_x", None)
        val = kwargs.get("val", None)
        test = kwargs.get("test", None)

        if "train_x" in kwargs and "val" in kwargs and "test" in kwargs:
            if _safe_to_merge(train_x, val, test):
                train_list = list(train_x)
                val_list = list(val)

                merged_train = train_list + val_list
                merged_val = [] if _drop_val_from_method_config() else val_list

                print(
                    "[KPlusValProtocol] MERGE few-shot val into train: "
                    f"train={len(train_list)} + val={len(val_list)} "
                    f"-> train={len(merged_train)}, "
                    f"val={'dropped' if len(merged_val) == 0 else len(merged_val)}"
                )

                kwargs["train_x"] = merged_train
                kwargs["val"] = merged_val

        return _ORIGINAL_DATASETBASE_INIT(self, *args, **kwargs)

    DatasetBase.__init__ = patched_init
    _PATCH_INSTALLED = True

    print("[KPlusValProtocol] DatasetBase patch installed from method config.")
