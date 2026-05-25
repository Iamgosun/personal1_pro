import os
import glob
from pathlib import Path

import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY

from .retina_common import (
    RetinaFundusBase,
    list_images,
    explode_categories,
    find_column,
    label_to_int,
)
from .oxford_pets import OxfordPets


MMAC_CLASSNAMES = [
    "no myopic maculopathy",
    "tessellated fundus",
    "diffuse chorioretinal atrophy",
    "patchy chorioretinal atrophy",
    "macular atrophy",
]
MMAC_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(MMAC_CLASSNAMES)}
MMAC_LABEL_TEXT_TO_INT = {
    "no myopic maculopathy": 0,
    "normal": 0,
    "tessellated fundus": 1,
    "diffuse chorioretinal atrophy": 2,
    "patchy chorioretinal atrophy": 3,
    "macular atrophy": 4,
}


@DATASET_REGISTRY.register()
class MMAC5(RetinaFundusBase):
    dataset_dir = "mmac"
    split_filename = "split_zhou_MMAC5.json"

    def read_data(self):
        train = self._read_mmac_split("classification_train", "train")
        val_all = self._read_mmac_split("classification_val", "val")

        val, test = OxfordPets.split_trainval(val_all, p_val=0.5)
        return train, val, test

    def _read_mmac_split(self, split_dir_name, inner_split_name):
        split_dir = os.path.join(self.dataset_dir, split_dir_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Missing MMAC split directory: {split_dir}")

        label_files = glob.glob(os.path.join(split_dir, "Groundtruths", "*.csv"))
        if len(label_files) == 0:
            raise FileNotFoundError(f"No MMAC label csv found under {split_dir}")

        df = pd.read_csv(label_files[0])

        image_root_candidates = [
            os.path.join(split_dir, "Images", inner_split_name),
            os.path.join(split_dir, "Images"),
        ]
        image_root = None
        for candidate in image_root_candidates:
            if os.path.exists(candidate):
                image_root = candidate
                break

        if image_root is None:
            raise FileNotFoundError(f"No MMAC image root found under {split_dir}")

        image_col = find_column(
            df,
            candidates=["image", "image_id", "filename", "file", "name"],
            contains=["image"],
        )
        label_col = find_column(
            df,
            candidates=["label", "class", "grade", "category"],
            contains=[],
        )

        images = list_images(image_root)
        stem_to_path = {Path(p).stem.lower(): p for p in images}

        items = []
        for _, row in df.iterrows():
            image_name = str(row[image_col]).strip()
            try:
                label = label_to_int(row[label_col], MMAC_LABEL_TEXT_TO_INT)
            except ValueError:
                continue

            if label < 0 or label >= len(MMAC_CLASSNAMES):
                continue

            impath = os.path.join(image_root, image_name)
            if not os.path.exists(impath):
                impath = stem_to_path.get(Path(image_name).stem.lower())
                if impath is None:
                    continue

            classname = MMAC_CLASSNAMES[label]
            items.extend(
                explode_categories(
                    impath=impath,
                    categories=[classname],
                    class_to_label=MMAC_CLASS_TO_LABEL,
                )
            )

        return items
