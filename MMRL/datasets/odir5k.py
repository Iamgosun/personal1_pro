import os

import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY

from .retina_common import (
    RetinaFundusBase,
    records_to_items,
    split_records_by_primary_label,
    unique_preserve_order,
    find_column,
)


ODIR_CLASSNAMES = [
    "normal fundus",
    "diabetic retinopathy",
    "glaucoma",
    "cataract",
    "age related macular degeneration",
    "hypertensive retinopathy",
    "pathological myopia",
    "other retinal abnormality",
]
ODIR_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(ODIR_CLASSNAMES)}

ODIR_CODE_TO_CLASSNAME = {
    "N": "normal fundus",
    "D": "diabetic retinopathy",
    "G": "glaucoma",
    "C": "cataract",
    "A": "age related macular degeneration",
    "H": "hypertensive retinopathy",
    "M": "pathological myopia",
    "O": "other retinal abnormality",
}


@DATASET_REGISTRY.register()
class ODIR5KSingle(RetinaFundusBase):
    """ODIR-5K with FLAIR-style multi-label-to-single-label expansion.

    Supported local layout:

        odir5k/
        ├── Training Images/
        ├── Testing Images/
        └── data.xlsx

    A multi-disease image is not discarded. It is expanded into multiple Datum
    objects, one per positive category.
    """

    dataset_dir = "odir5k"
    split_filename = "split_zhou_ODIR5KSingle.json"

    anno_candidates = [
        "data.xlsx",
        "ODIR-5K_Training_Annotations (Updated)_V2.xlsx",
        "ODIR-5K_Training_Annotations(Updated)_V2.xlsx",
        "ODIR-5K_Training_Annotations.xlsx",
        "Training_Annotations.xlsx",
    ]

    def read_data(self):
        anno_path = self._find_annotation_file()

        image_dir = os.path.join(self.dataset_dir, "Training Images")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Missing ODIR training image directory: {image_dir}")

        df = pd.read_excel(anno_path)
        print(f"[ODIR5KSingle] annotation file: {anno_path}")
        print(f"[ODIR5KSingle] columns: {list(df.columns)}")

        left_image_col = self._find_side_image_col(df, "Left")
        right_image_col = self._find_side_image_col(df, "Right")
        left_kw_col = self._find_side_keyword_col(df, "Left")
        right_kw_col = self._find_side_keyword_col(df, "Right")

        if left_image_col is None and right_image_col is None:
            raise KeyError(
                "Cannot find ODIR image columns. Expected columns like "
                "'Left-Fundus' and 'Right-Fundus'."
            )

        records = []
        dropped_no_label = 0
        dropped_no_image = 0

        for _, row in df.iterrows():
            row_fallback_categories = self._categories_from_onehot_columns(row)

            side_specs = [
                ("Left", left_image_col, left_kw_col),
                ("Right", right_image_col, right_kw_col),
            ]

            for side, image_col, kw_col in side_specs:
                if image_col is None:
                    continue

                image_name = row.get(image_col)
                if not isinstance(image_name, str) or len(image_name.strip()) == 0:
                    continue

                impath = os.path.join(image_dir, image_name)
                if not os.path.exists(impath):
                    dropped_no_image += 1
                    continue

                categories = self._categories_from_keyword_col(row, kw_col)
                if len(categories) == 0:
                    categories = row_fallback_categories

                categories = self._clean_categories(categories)
                if len(categories) == 0:
                    dropped_no_label += 1
                    continue

                records.append({"impath": impath, "categories": categories, "side": side})

        print(
            f"[ODIR5KSingle] image-level records={len(records)}, "
            f"dropped_no_label={dropped_no_label}, dropped_no_image={dropped_no_image}"
        )

        train_records, val_records, test_records = split_records_by_primary_label(records)

        train = records_to_items(train_records, ODIR_CLASS_TO_LABEL)
        val = records_to_items(val_records, ODIR_CLASS_TO_LABEL)
        test = records_to_items(test_records, ODIR_CLASS_TO_LABEL)

        print(
            f"[ODIR5KSingle] exploded Datum: "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

        return train, val, test

    def _find_annotation_file(self):
        for name in self.anno_candidates:
            path = os.path.join(self.dataset_dir, name)
            if os.path.exists(path):
                return path

        xlsx_files = [
            os.path.join(self.dataset_dir, f)
            for f in os.listdir(self.dataset_dir)
            if f.lower().endswith((".xlsx", ".xls"))
        ]
        if len(xlsx_files) == 1:
            return xlsx_files[0]

        raise FileNotFoundError(
            "Missing ODIR annotation xlsx. Tried: "
            + ", ".join(os.path.join(self.dataset_dir, n) for n in self.anno_candidates)
        )

    @staticmethod
    def _find_side_image_col(df, side):
        candidates = [
            f"{side}-Fundus",
            f"{side} Fundus",
            f"{side}_Fundus",
            f"{side.lower()}-fundus",
            f"{side.lower()} fundus",
            f"{side.lower()}_fundus",
        ]
        return find_column(
            df,
            candidates=candidates,
            contains=[side.lower(), "fundus"],
            required=False,
        )

    @staticmethod
    def _find_side_keyword_col(df, side):
        candidates = [
            f"{side}-Diagnostic Keywords",
            f"{side} Diagnostic Keywords",
            f"{side}_Diagnostic Keywords",
            f"{side.lower()}-diagnostic keywords",
            f"{side.lower()} diagnostic keywords",
            f"{side.lower()}_diagnostic keywords",
        ]
        return find_column(
            df,
            candidates=candidates,
            contains=[side.lower(), "diagnostic"],
            required=False,
        )

    def _categories_from_onehot_columns(self, row):
        categories = []
        for code, classname in ODIR_CODE_TO_CLASSNAME.items():
            if code not in row:
                continue
            try:
                value = int(row[code])
            except Exception:
                value = 0
            if value == 1:
                categories.append(classname)
        return self._clean_categories(categories)

    def _categories_from_keyword_col(self, row, keyword_col):
        if keyword_col is None:
            return []

        text = row.get(keyword_col)
        if not isinstance(text, str):
            return []

        return self._categories_from_keyword_text(text)

    @staticmethod
    def _categories_from_keyword_text(text):
        text = str(text).lower()
        categories = []

        if "normal fundus" in text or text.strip() == "normal":
            categories.append("normal fundus")

        # Order matters: hypertensive retinopathy must not be captured as diabetic retinopathy.
        if "hypertensive retinopathy" in text or "hypertension" in text:
            categories.append("hypertensive retinopathy")

        # ODIR keywords often use "non proliferative retinopathy" for DR.
        if (
            "diabetic retinopathy" in text
            or "non proliferative retinopathy" in text
            or "non-proliferative retinopathy" in text
            or "proliferative retinopathy" in text
            or "proliferative diabetic retinopathy" in text
            or text.strip() == "dr"
        ):
            categories.append("diabetic retinopathy")

        if "glaucoma" in text:
            categories.append("glaucoma")
        if "cataract" in text:
            categories.append("cataract")
        if "age-related macular degeneration" in text or "age related macular degeneration" in text or "amd" in text:
            categories.append("age related macular degeneration")
        if "pathological myopia" in text or "pathologic myopia" in text:
            categories.append("pathological myopia")

        if len(categories) == 0 and len(text.strip()) > 0:
            categories.append("other retinal abnormality")

        return categories

    @staticmethod
    def _clean_categories(categories):
        categories = unique_preserve_order(categories)
        disease = [c for c in categories if c != "normal fundus"]
        if len(disease) > 0:
            return disease
        return categories
