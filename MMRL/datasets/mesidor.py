import os
import glob
from pathlib import Path

import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY

from .retina_common import (
    RetinaFundusBase,
    list_images,
    make_item,
    records_to_items,
    split_records_by_primary_label,
    find_column,
    label_to_int,
)


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
class MESIDORDR5(RetinaFundusBase):
    dataset_dir = "mesidor"
    split_filename = "split_zhou_MESIDORDR5.json"

    def read_data(self):
        items = []

        base_dirs = [
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
            and d.lower().startswith("base")
        ]
        base_dirs.sort()

        if len(base_dirs) == 0:
            raise RuntimeError(f"No Base* folders found under {self.dataset_dir}")

        for base in base_dirs:
            base_dir = os.path.join(self.dataset_dir, base)
            label_files = glob.glob(os.path.join(base_dir, "*.xls*"))

            if len(label_files) == 0:
                print(f"[MESIDORDR5] WARNING: no Excel label file under {base_dir}")
                continue

            df = pd.read_excel(label_files[0])
            items.extend(self._read_base(df, base_dir))

        records = [{"impath": item.impath, "categories": [item.classname]} for item in items]
        train_records, val_records, test_records = split_records_by_primary_label(records)
        return (
            records_to_items(train_records, DR5_CLASS_TO_LABEL),
            records_to_items(val_records, DR5_CLASS_TO_LABEL),
            records_to_items(test_records, DR5_CLASS_TO_LABEL),
        )

    def _read_base(self, df, base_dir):
        image_col = find_column(
            df,
            candidates=["image", "image name", "filename", "file", "name"],
            contains=["image"],
        )
        grade_col = find_column(
            df,
            candidates=["retinopathy grade", "dr", "grade", "retinopathy"],
            contains=["grade"],
        )

        images = list_images(base_dir)
        stem_to_path = {Path(p).stem.lower(): p for p in images}

        items = []
        for _, row in df.iterrows():
            image_name = str(row[image_col]).strip()
            try:
                label = label_to_int(row[grade_col], DR_LABEL_TEXT_TO_INT)
            except ValueError:
                continue

            if label < 0 or label >= len(DR5_CLASSNAMES):
                continue

            impath = os.path.join(base_dir, image_name)
            if not os.path.exists(impath):
                impath = stem_to_path.get(Path(image_name).stem.lower())
                if impath is None:
                    continue

            classname = DR5_CLASSNAMES[label]
            items.append(make_item(impath, label, classname))

        return items
