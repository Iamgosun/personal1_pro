import os
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


DR5_CLASSNAMES = [
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
]
DR5_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(DR5_CLASSNAMES)}
DR_LABEL_TEXT_TO_INT = {
    "no diabetic retinopathy": 0,
    "no dr": 0,
    "normal": 0,
    "mild diabetic retinopathy": 1,
    "mild": 1,
    "moderate diabetic retinopathy": 2,
    "moderate": 2,
    "severe diabetic retinopathy": 3,
    "severe": 3,
    "proliferative diabetic retinopathy": 4,
    "pdr": 4,
    "proliferative": 4,
}


@DATASET_REGISTRY.register()
class DeepDRiD5(RetinaFundusBase):
    dataset_dir = "deepdrid"
    split_filename = "split_zhou_DeepDRiD5.json"

    def read_data(self):
        train_csv = os.path.join(
            self.dataset_dir,
            "regular_fundus_images",
            "regular-fundus-training",
            "regular-fundus-training.csv",
        )
        val_csv = os.path.join(
            self.dataset_dir,
            "regular_fundus_images",
            "regular-fundus-validation",
            "regular-fundus-validation.csv",
        )

        train = self._read_csv_split(
            csv_path=train_csv,
            image_root=os.path.join(
                self.dataset_dir,
                "regular_fundus_images",
                "regular-fundus-training",
                "Images",
            ),
        )
        val_all = self._read_csv_split(
            csv_path=val_csv,
            image_root=os.path.join(
                self.dataset_dir,
                "regular_fundus_images",
                "regular-fundus-validation",
                "Images",
            ),
        )

        val, test = OxfordPets.split_trainval(val_all, p_val=0.5)
        return train, val, test

    def _read_csv_split(self, csv_path, image_root):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing DeepDRiD label csv: {csv_path}")
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Missing DeepDRiD image directory: {image_root}")

        df = pd.read_csv(csv_path)

        image_col = find_column(
            df,
            candidates=["image", "image_id", "filename", "file", "img"],
            contains=["image"],
        )
        label_col = find_column(
            df,
            candidates=["label", "dr", "grade", "retinopathy_grade", "DR_grade"],
            contains=["grade"],
        )

        images = list_images(image_root)
        stem_to_path = {Path(p).stem.lower(): p for p in images}

        items = []
        for _, row in df.iterrows():
            image_name = str(row[image_col]).strip()
            label = label_to_int(row[label_col], DR_LABEL_TEXT_TO_INT)

            if label < 0 or label >= len(DR5_CLASSNAMES):
                continue

            impath = os.path.join(image_root, image_name)
            if not os.path.exists(impath):
                impath = stem_to_path.get(Path(image_name).stem.lower())
                if impath is None:
                    continue

            classname = DR5_CLASSNAMES[label]
            items.extend(
                explode_categories(
                    impath=impath,
                    categories=[classname],
                    class_to_label=DR5_CLASS_TO_LABEL,
                )
            )

        return items
