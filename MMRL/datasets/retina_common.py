import os
import glob
import pickle
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

from dassl.data.datasets import Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".WEBP",
)


def list_images(root):
    root = os.path.abspath(os.path.expanduser(root))
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(set(paths))


def make_item(impath, label, classname):
    return Datum(impath=str(impath), label=int(label), classname=str(classname))


def unique_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def explode_categories(impath, categories, class_to_label, normal_classname="normal fundus"):
    """FLAIR-style multi-label -> single-label Datum expansion.

    FLAIR keeps categories as a list and samples one textual category at
    __getitem__ time. This repo materializes Datum(impath, label, classname),
    so the compatible equivalent is static expansion:

        image + [cat_a, cat_b]
        -> Datum(image, label(cat_a), cat_a)
        -> Datum(image, label(cat_b), cat_b)
    """
    categories = unique_preserve_order(categories)

    non_normal = [c for c in categories if c != normal_classname]
    if len(non_normal) > 0:
        categories = non_normal

    items = []
    for classname in categories:
        if classname not in class_to_label:
            continue
        items.append(make_item(impath, class_to_label[classname], classname))
    return items


def split_records_by_primary_label(records, p_train=0.5, p_val=0.2):
    """Split at image/record level before explosion to avoid image leakage."""
    buckets = defaultdict(list)
    for record in records:
        categories = record.get("categories", [])
        primary = categories[0] if categories else "__unknown__"
        buckets[primary].append(record)

    train, val, test = [], [], []
    for _, bucket in buckets.items():
        random.shuffle(bucket)
        n_total = len(bucket)
        if n_total < 3:
            train.extend(bucket)
            continue

        n_train = round(n_total * p_train)
        n_val = round(n_total * p_val)

        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    return train, val, test


def records_to_items(records, class_to_label):
    items = []
    for record in records:
        items.extend(
            explode_categories(
                impath=record["impath"],
                categories=record["categories"],
                class_to_label=class_to_label,
            )
        )
    return items


def find_column(df, candidates=None, contains=None, required=True):
    candidates = candidates or []
    contains = contains or []

    columns = list(df.columns)
    normalized = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]

    for column in columns:
        text = str(column).strip().lower()
        if all(term.lower() in text for term in contains):
            return column

    if required:
        raise KeyError(
            f"Cannot find column. candidates={candidates}, contains={contains}, columns={columns}"
        )

    return None


def label_to_int(value, mapping=None):
    if mapping is not None:
        text = str(value).strip().lower()
        if text in mapping:
            return int(mapping[text])

    if pd.isna(value):
        raise ValueError("Cannot convert NaN label to int")

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            raise ValueError("Cannot convert empty label to int")
        try:
            return int(float(value))
        except ValueError as exc:
            raise ValueError(f"Cannot convert label to int: {value}") from exc

    return int(value)


class RetinaFundusBase(DatasetBase):
    """Base class matching the existing natural-image datasets in this repo."""

    dataset_dir = None
    split_filename = None

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, self.split_filename)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            train, val, test = self.read_data()

            if len(val) == 0:
                train, val = OxfordPets.split_trainval(train)

            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-seed_{seed}.pkl",
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, val, test = OxfordPets.subsample_classes(
            train,
            val,
            test,
            subsample=cfg.DATASET.SUBSAMPLE_CLASSES,
        )

        print(
            f"[{self.__class__.__name__}] train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self):
        raise NotImplementedError
