import os
import re
import glob
import pickle
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP",
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
    output = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def explode_categories(impath, categories, class_to_label, normal_classname="normal fundus"):
    """FLAIR-style multi-label category list -> single-label Datum expansion.

    Example:
        image + [cat_a, cat_b]
        -> Datum(image, label(cat_a), cat_a)
        -> Datum(image, label(cat_b), cat_b)

    This keeps this repository's single-label DatasetBase/Datum pipeline.
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
    """Split image-level records before category expansion to avoid leakage."""
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

    if contains:
        for column in columns:
            text = str(column).strip().lower()
            if all(term.lower() in text for term in contains):
                return column

    if required:
        raise KeyError(
            f"Cannot find column. candidates={candidates}, "
            f"contains={contains}, columns={columns}"
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


def resolve_dataset_dir(root, candidates):
    tried = []
    for name in candidates:
        path = os.path.join(root, name)
        tried.append(path)
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        "Cannot find dataset directory. Tried:\n  " + "\n  ".join(tried)
    )


class RetinaFundusBase(DatasetBase):
    dataset_dir = None
    dataset_dir_candidates = None
    split_filename = None

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))

        if self.dataset_dir_candidates is not None:
            self.dataset_dir = resolve_dataset_dir(root, self.dataset_dir_candidates)
        else:
            self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.split_path = os.path.join(self.dataset_dir, self.split_filename)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            train, val, test = self.read_data()

            if len(train) == 0:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] train split has 0 samples. "
                    f"Check dataset_dir={self.dataset_dir} and parser logs above."
                )

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
            train, val, test, subsample=cfg.DATASET.SUBSAMPLE_CLASSES
        )

        print(f"[{self.__class__.__name__}] dataset_dir={self.dataset_dir}")
        print(
            f"[{self.__class__.__name__}] train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FIVES
# ---------------------------------------------------------------------------

FIVES_CLASSNAMES = [
    "age related macular degeneration",
    "diabetic retinopathy",
    "glaucoma",
    "normal fundus",
]
FIVES_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(FIVES_CLASSNAMES)}
FIVES_SUFFIX_TO_CLASSNAME = {
    "A": "age related macular degeneration",
    "D": "diabetic retinopathy",
    "G": "glaucoma",
    "N": "normal fundus",
}


@DATASET_REGISTRY.register()
class FIVES4(RetinaFundusBase):
    dataset_dir_candidates = ["fives", "FIVES"]
    split_filename = "split_zhou_FIVES4.json"

    def read_data(self):
        train_dir = os.path.join(self.dataset_dir, "train", "Original")
        test_dir = os.path.join(self.dataset_dir, "test", "Original")

        trainval = self._read_image_dir(train_dir)
        test = self._read_image_dir(test_dir)

        train, val = OxfordPets.split_trainval(trainval)
        return train, val, test

    def _read_image_dir(self, image_dir):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"FIVES image directory not found: {image_dir}")

        items = []
        for impath in list_images(image_dir):
            stem = Path(impath).stem
            suffix = stem.split("_")[-1]
            if suffix not in FIVES_SUFFIX_TO_CLASSNAME:
                suffix = stem[-1]

            if suffix not in FIVES_SUFFIX_TO_CLASSNAME:
                raise ValueError(f"Cannot infer FIVES label from filename: {impath}")

            classname = FIVES_SUFFIX_TO_CLASSNAME[suffix]
            items.extend(explode_categories(impath, [classname], FIVES_CLASS_TO_LABEL))

        return items


# ---------------------------------------------------------------------------
# 1000x39
# ---------------------------------------------------------------------------

FUNDUS1000X39_CNAMES = {
    "0.0.Normal": "normal fundus",
    "0.1.Tessellated fundus": "tessellated fundus",
    "0.2.Large optic cup": "large optic cup",
    "0.3.DR1": "mild diabetic retinopathy",
    "1.0.DR2": "moderate diabetic retinopathy",
    "1.1.DR3": "severe diabetic retinopathy",
    "2.0.BRVO": "branch retinal vein occlusion",
    "2.1.CRVO": "central retinal vein occlusion",
    "3.RAO": "retinal artery occlusion",
    "4.Rhegmatogenous RD": "rhegmatogenous retinal detachment",
    "5.0.CSCR": "central serous chorioretinopathy",
    "5.1.VKH disease": "Vogt Koyanagi Harada disease",
    "6.Maculopathy": "maculopathy",
    "7.ERM": "epiretinal membrane",
    "8.MH": "macular hole",
    "9.Pathological myopia": "pathological myopia",
    "10.0.Possible glaucoma": "possible glaucoma",
    "10.1.Optic atrophy": "optic atrophy",
    "11.Severe hypertensive retinopathy": "severe hypertensive retinopathy",
    "12.Disc swelling and elevation": "disc swelling and elevation",
    "13.Dragged Disc": "dragged disc",
    "14.Congenital disc abnormality": "congenital disc abnormality",
    "15.0.Retinitis pigmentosa": "retinitis pigmentosa",
    "15.1.Bietti crystalline dystrophy": "Bietti crystalline dystrophy",
    "16.Peripheral retinal degeneration and break": "peripheral retinal degeneration and break",
    "17.Myelinated nerve fiber": "myelinated nerve fiber",
    "18.Vitreous particles": "vitreous particles",
    "19.Fundus neoplasm": "fundus neoplasm",
    "20.Massive hard exudates": "massive hard exudates",
    "21.Yellow-white spots-flecks": "yellow white spots and flecks",
    "22.Cotton-wool spots": "cotton wool spots",
    "23.Vessel tortuosity": "vessel tortuosity",
    "24.Chorioretinal atrophy-coloboma": "chorioretinal atrophy or coloboma",
    "25.Preretinal hemorrhage": "preretinal hemorrhage",
    "26.Fibrosis": "fibrosis",
    "27.Laser Spots": "laser scars",
    "28.Silicon oil in eye": "silicone oil in eye",
    "29.0.Blur fundus without PDR": "blurred fundus without proliferative diabetic retinopathy",
    "29.1.Blur fundus with suspected PDR": "blurred fundus with suspected proliferative diabetic retinopathy",
}
FUNDUS1000X39_CLASSNAMES = [
    FUNDUS1000X39_CNAMES[k] for k in sorted(FUNDUS1000X39_CNAMES.keys())
]
FUNDUS1000X39_CLASS_TO_LABEL = {
    name: idx for idx, name in enumerate(FUNDUS1000X39_CLASSNAMES)
}


@DATASET_REGISTRY.register()
class Fundus1000x39(RetinaFundusBase):
    dataset_dir_candidates = ["1000x39", "Fundus1000x39", "fundus1000x39"]
    split_filename = "split_zhou_Fundus1000x39.json"

    def read_data(self):
        categories = [
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ]
        categories = [c for c in categories if c in FUNDUS1000X39_CNAMES]
        categories.sort()

        if len(categories) == 0:
            raise RuntimeError(
                f"No recognized 1000x39 category folders found under {self.dataset_dir}"
            )

        records = []
        for category in categories:
            classname = FUNDUS1000X39_CNAMES[category]
            for impath in list_images(os.path.join(self.dataset_dir, category)):
                records.append({"impath": impath, "categories": [classname]})

        train_records, val_records, test_records = split_records_by_primary_label(records)
        return (
            records_to_items(train_records, FUNDUS1000X39_CLASS_TO_LABEL),
            records_to_items(val_records, FUNDUS1000X39_CLASS_TO_LABEL),
            records_to_items(test_records, FUNDUS1000X39_CLASS_TO_LABEL),
        )


# ---------------------------------------------------------------------------
# ODIR-5K
# ---------------------------------------------------------------------------

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
    dataset_dir_candidates = ["odir5k", "ODIR-5K", "ODIR5K"]
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

            for side, image_col, kw_col in [
                ("Left", left_image_col, left_kw_col),
                ("Right", right_image_col, right_kw_col),
            ]:
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
        return find_column(
            df,
            candidates=[
                f"{side}-Fundus",
                f"{side} Fundus",
                f"{side}_Fundus",
                f"{side.lower()}-fundus",
                f"{side.lower()} fundus",
                f"{side.lower()}_fundus",
            ],
            contains=[side.lower(), "fundus"],
            required=False,
        )

    @staticmethod
    def _find_side_keyword_col(df, side):
        return find_column(
            df,
            candidates=[
                f"{side}-Diagnostic Keywords",
                f"{side} Diagnostic Keywords",
                f"{side}_Diagnostic Keywords",
                f"{side.lower()}-diagnostic keywords",
                f"{side.lower()} diagnostic keywords",
                f"{side.lower()}_diagnostic keywords",
            ],
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

        if "hypertensive retinopathy" in text or "hypertension" in text:
            categories.append("hypertensive retinopathy")

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
        if (
            "age-related macular degeneration" in text
            or "age related macular degeneration" in text
            or "amd" in text
        ):
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


# ---------------------------------------------------------------------------
# DR5 class definitions for Kaggle DeepDRiD and Messidor2Preprocess
# ---------------------------------------------------------------------------

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
    """Parser for the Kaggle yoctoman/deepdrid layout.

    Expected layout:
        deepdrid/
        └── regular_fundus_images/
            ├── regular-fundus-training/
            │   ├── Images/
            │   └── regular-fundus-training.csv
            └── regular-fundus-validation/
                ├── Images/
                └── regular-fundus-validation.csv

    Observed CSV columns:
        patient_id,image_id,image_path,Overall quality,
        left_eye_DR_Level,right_eye_DR_Level,patient_DR_Level,
        Clarity,Field definition,Artifact

    Observed CSV image_path:
        regular-fundus-validation\\265\\265_l1.jpg

    Real image path:
        regular-fundus-validation/Images/265/265_l1.jpg
    """

    dataset_dir_candidates = ["deepdrid", "DeepDRiD", "deepdird", "DeepDRiD5"]
    split_filename = "split_zhou_DeepDRiD5.json"

    def read_data(self):
        train_root = os.path.join(
            self.dataset_dir,
            "regular_fundus_images",
            "regular-fundus-training",
        )
        val_root = os.path.join(
            self.dataset_dir,
            "regular_fundus_images",
            "regular-fundus-validation",
        )

        train_csv = os.path.join(train_root, "regular-fundus-training.csv")
        val_csv = os.path.join(val_root, "regular-fundus-validation.csv")

        train = self._read_csv_split(train_csv, train_root, "regular-fundus-training")
        val_all = self._read_csv_split(val_csv, val_root, "regular-fundus-validation")

        if len(train) == 0:
            raise RuntimeError(
                "[DeepDRiD5] train loaded 0 samples. "
                "Check printed sample_csv_image_values and sample_image_files."
            )

        if len(val_all) == 0:
            raise RuntimeError(
                "[DeepDRiD5] validation loaded 0 samples. "
                "Check printed sample_csv_image_values and sample_image_files."
            )

        val, test = OxfordPets.split_trainval(val_all, p_val=0.5)
        return train, val, test

    def _read_csv_split(self, csv_path, split_root, split_name):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing DeepDRiD csv: {csv_path}")
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Missing DeepDRiD split root: {split_root}")

        df = pd.read_csv(csv_path)

        print(f"[DeepDRiD5] csv={csv_path}")
        print(f"[DeepDRiD5] split_root={split_root}")
        print(f"[DeepDRiD5] columns={list(df.columns)}")

        image_col = find_column(
            df,
            candidates=["image_path", "image", "filename", "file", "img", "image_id"],
            contains=["image"],
        )
        image_id_col = find_column(
            df,
            candidates=["image_id", "id"],
            contains=["image", "id"],
            required=False,
        )

        left_label_col = find_column(
            df,
            candidates=["left_eye_DR_Level", "left_eye_dr_level", "left dr level"],
            contains=["left", "dr"],
            required=False,
        )
        right_label_col = find_column(
            df,
            candidates=["right_eye_DR_Level", "right_eye_dr_level", "right dr level"],
            contains=["right", "dr"],
            required=False,
        )
        patient_label_col = find_column(
            df,
            candidates=["patient_DR_Level", "patient_dr_level", "patient dr level"],
            contains=["patient", "dr"],
            required=False,
        )
        generic_label_col = find_column(
            df,
            candidates=["label", "dr", "grade", "retinopathy_grade", "DR_grade"],
            contains=["grade"],
            required=False,
        )

        print(f"[DeepDRiD5] image_col={image_col}")
        print(f"[DeepDRiD5] image_id_col={image_id_col}")
        print(f"[DeepDRiD5] left_label_col={left_label_col}")
        print(f"[DeepDRiD5] right_label_col={right_label_col}")
        print(f"[DeepDRiD5] patient_label_col={patient_label_col}")
        print(f"[DeepDRiD5] generic_label_col={generic_label_col}")

        image_roots = [
            os.path.join(split_root, "Images"),
            os.path.join(split_root, "images"),
            split_root,
        ]
        image_roots = [p for p in image_roots if os.path.isdir(p)]

        all_images = []
        for root in image_roots:
            all_images.extend(list_images(root))
        all_images = sorted(set(all_images))

        print(f"[DeepDRiD5] image_roots={image_roots}")
        print(f"[DeepDRiD5] num_image_files={len(all_images)}")
        print(f"[DeepDRiD5] sample_csv_image_values={df[image_col].dropna().astype(str).head(5).tolist()}")
        print(f"[DeepDRiD5] sample_image_files={all_images[:5]}")

        image_index = self._build_image_index(all_images, image_roots, split_name)

        items = []
        skipped_no_label = 0
        skipped_bad_label = 0
        skipped_no_image = 0

        for _, row in df.iterrows():
            image_value = str(row[image_col]).strip()

            image_id_value = None
            if image_id_col is not None:
                image_id_value = str(row[image_id_col]).strip()

            label = self._get_deepdrid_label(
                row=row,
                image_value=image_value,
                image_id_value=image_id_value,
                left_label_col=left_label_col,
                right_label_col=right_label_col,
                patient_label_col=patient_label_col,
                generic_label_col=generic_label_col,
            )

            if label is None:
                skipped_no_label += 1
                continue

            if label < 0 or label >= len(DR5_CLASSNAMES):
                skipped_bad_label += 1
                continue

            impath = self._resolve_image_path(
                image_value=image_value,
                image_id_value=image_id_value,
                image_roots=image_roots,
                image_index=image_index,
                split_name=split_name,
            )

            if impath is None:
                skipped_no_image += 1
                continue

            classname = DR5_CLASSNAMES[label]
            items.extend(explode_categories(impath, [classname], DR5_CLASS_TO_LABEL))

        print(
            f"[DeepDRiD5] loaded={len(items)}, "
            f"skipped_no_label={skipped_no_label}, "
            f"skipped_bad_label={skipped_bad_label}, "
            f"skipped_no_image={skipped_no_image}"
        )

        return items

    @staticmethod
    def _norm_path(value):
        value = str(value).strip().replace("\\\\", "/").replace("\\", "/")
        while value.startswith("./"):
            value = value[2:]
        return value

    def _normalize_key(self, value):
        return self._norm_path(value).lower()

    def _build_image_index(self, all_images, image_roots, split_name):
        index = {}

        for path in all_images:
            p = Path(path)
            keys = set()

            keys.add(self._normalize_key(path))
            keys.add(self._normalize_key(p.name))
            keys.add(self._normalize_key(p.stem))

            for root in image_roots:
                try:
                    rel = os.path.relpath(path, root)
                except ValueError:
                    continue

                rel = self._norm_path(rel)
                keys.add(self._normalize_key(rel))
                keys.add(self._normalize_key(str(Path(rel).with_suffix(""))))

                keys.add(self._normalize_key(f"Images/{rel}"))
                keys.add(self._normalize_key(f"images/{rel}"))
                keys.add(self._normalize_key(f"{split_name}/{rel}"))
                keys.add(self._normalize_key(f"{split_name}/Images/{rel}"))
                keys.add(self._normalize_key(f"{split_name}/images/{rel}"))

            for key in keys:
                if key and key not in index:
                    index[key] = path

        return index

    def _get_deepdrid_label(
        self,
        row,
        image_value,
        image_id_value,
        left_label_col,
        right_label_col,
        patient_label_col,
        generic_label_col,
    ):
        text = f"{image_value} {image_id_value or ''}".lower()

        def parse_col(col):
            if col is None:
                return None
            try:
                value = row[col]
                if pd.isna(value):
                    return None
                if isinstance(value, str) and value.strip() == "":
                    return None
                return label_to_int(value, DR_LABEL_TEXT_TO_INT)
            except Exception:
                return None

        if "_l" in text or "-l" in text or "/l" in text:
            label = parse_col(left_label_col)
            if label is not None:
                return label

        if "_r" in text or "-r" in text or "/r" in text:
            label = parse_col(right_label_col)
            if label is not None:
                return label

        for col in [patient_label_col, generic_label_col, left_label_col, right_label_col]:
            label = parse_col(col)
            if label is not None:
                return label

        return None

    def _resolve_image_path(
        self,
        image_value,
        image_id_value,
        image_roots,
        image_index,
        split_name,
    ):
        raw_values = []

        if image_value is not None:
            raw_values.append(str(image_value).strip())

        if image_id_value is not None:
            image_id_value = str(image_id_value).strip()
            if image_id_value and image_id_value not in raw_values:
                raw_values.append(image_id_value)

        for raw in raw_values:
            raw = self._norm_path(raw)
            if not raw:
                continue

            variants = []
            variants.append(raw)

            parts = raw.split("/")
            if len(parts) >= 2 and parts[0].lower() in {
                "regular-fundus-training",
                "regular-fundus-validation",
                "regular_fundus_training",
                "regular_fundus_validation",
            }:
                stripped = "/".join(parts[1:])
                variants.append(stripped)
                variants.append(f"Images/{stripped}")
                variants.append(f"images/{stripped}")

            variants.append(f"Images/{raw}")
            variants.append(f"images/{raw}")
            variants.append(os.path.basename(raw))
            variants.append(Path(raw).stem)

            dedup = []
            seen = set()
            for v in variants:
                v = self._norm_path(v)
                if v and v not in seen:
                    seen.add(v)
                    dedup.append(v)

            for v in dedup:
                for root in image_roots:
                    candidates = [
                        v,
                        os.path.join(root, v),
                        os.path.join(root, os.path.basename(v)),
                        os.path.join(os.path.dirname(root), v),
                        os.path.join(os.path.dirname(root), os.path.basename(v)),
                    ]
                    for candidate in candidates:
                        candidate = self._norm_path(candidate)
                        if os.path.exists(candidate):
                            return candidate

            for v in dedup:
                keys = [
                    self._normalize_key(v),
                    self._normalize_key(os.path.basename(v)),
                    self._normalize_key(Path(v).stem),
                    self._normalize_key(str(Path(v).with_suffix(""))),
                ]

                for key in keys:
                    if key in image_index:
                        return image_index[key]

            for v in dedup:
                key = self._normalize_key(str(Path(v).with_suffix("")))
                for known_key, path in image_index.items():
                    if known_key == key or known_key.endswith(key):
                        return path

        return None


@DATASET_REGISTRY.register()
class MESSIDORDR5(RetinaFundusBase):
    """Parser for the Kaggle mariaherrerot/messidor2preprocess layout.

    Expected layout:
        mesidor/
        ├── preprocess/
        ├── split_fewshot/
        └── messidor_data.csv
    """

    dataset_dir_candidates = ["mesidor", "MESIDOR", "messidor", "MESSIDOR", "Messidor"]
    split_filename = "split_zhou_MESSIDORDR5.json"

    csv_candidates = [
        "messidor_data.csv",
        "mesidor_data.csv",
        "MESSIDOR_data.csv",
        "metadata.csv",
        "metadata_dr5.csv",
        "labels.csv",
        "annotations.csv",
    ]

    image_dir_candidates = [
        "preprocess",
        "Preprocess",
        "processed",
        "Processed",
        "images",
        "Images",
        ".",
    ]

    def read_data(self):
        csv_path = self._find_csv_file()
        image_root = self._find_image_root()

        df = pd.read_csv(csv_path)
        print(f"[MESSIDORDR5] csv={csv_path}")
        print(f"[MESSIDORDR5] image_root={image_root}")
        print(f"[MESSIDORDR5] columns={list(df.columns)}")

        image_col = self._find_image_column(df)
        label_col = self._find_label_column(df)

        print(f"[MESSIDORDR5] image_col={image_col}")
        print(f"[MESSIDORDR5] label_col={label_col}")

        images = list_images(image_root)
        if len(images) == 0 and image_root != self.dataset_dir:
            images = list_images(self.dataset_dir)

        basename_to_path = {Path(p).name.lower(): p for p in images}
        stem_to_path = {Path(p).stem.lower(): p for p in images}

        records = []
        skipped_no_label = 0
        skipped_no_image = 0

        for _, row in df.iterrows():
            try:
                label = self._extract_label(row, label_col)
            except Exception:
                skipped_no_label += 1
                continue

            if label < 0 or label >= len(DR5_CLASSNAMES):
                skipped_no_label += 1
                continue

            impath = self._resolve_image_path(
                row=row,
                image_col=image_col,
                image_root=image_root,
                basename_to_path=basename_to_path,
                stem_to_path=stem_to_path,
            )

            if impath is None:
                skipped_no_image += 1
                continue

            classname = DR5_CLASSNAMES[label]
            records.append({"impath": impath, "categories": [classname]})

        print(
            f"[MESSIDORDR5] records={len(records)}, "
            f"skipped_no_label={skipped_no_label}, "
            f"skipped_no_image={skipped_no_image}"
        )

        if len(records) == 0:
            raise RuntimeError(
                "[MESSIDORDR5] No usable samples found. "
                "Check printed columns/image_col/label_col."
            )

        train_records, val_records, test_records = split_records_by_primary_label(records)

        return (
            records_to_items(train_records, DR5_CLASS_TO_LABEL),
            records_to_items(val_records, DR5_CLASS_TO_LABEL),
            records_to_items(test_records, DR5_CLASS_TO_LABEL),
        )

    def _find_csv_file(self):
        for name in self.csv_candidates:
            path = os.path.join(self.dataset_dir, name)
            if os.path.exists(path):
                return path

        csv_files = [
            os.path.join(self.dataset_dir, f)
            for f in os.listdir(self.dataset_dir)
            if f.lower().endswith(".csv")
        ]
        if len(csv_files) == 1:
            return csv_files[0]

        raise FileNotFoundError(
            "Cannot find MESSIDOR csv file. Tried:\n  "
            + "\n  ".join(os.path.join(self.dataset_dir, n) for n in self.csv_candidates)
        )

    def _find_image_root(self):
        for name in self.image_dir_candidates:
            path = os.path.join(self.dataset_dir, name)
            if os.path.isdir(path):
                return path
        return self.dataset_dir

    @staticmethod
    def _find_image_column(df):
        col = find_column(
            df,
            candidates=[
                "image_path",
                "path",
                "filepath",
                "file_path",
                "filename",
                "file",
                "image",
                "image_id",
                "id_code",
                "id",
                "name",
                "img",
            ],
            contains=["image"],
            required=False,
        )
        if col is not None:
            return col

        col = find_column(df, candidates=[], contains=["file"], required=False)
        if col is not None:
            return col

        for c in df.columns:
            values = df[c].dropna().astype(str).head(20).tolist()
            if any(
                v.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
                for v in values
            ):
                return c

        raise KeyError(f"Cannot find MESSIDOR image column. columns={list(df.columns)}")

    @staticmethod
    def _find_label_column(df):
        candidates = [
            "diagnosis",
            "adjudicated_dr_grade",
            "dr_grade",
            "DR_grade",
            "retinopathy_grade",
            "Retinopathy grade",
            "retinopathy grade",
            "grade",
            "level",
            "label",
            "class",
            "target",
        ]

        col = find_column(df, candidates=candidates, contains=[], required=False)
        if col is not None:
            return col

        for c in df.columns:
            text = str(c).strip().lower()
            if "retinopathy" in text and "grade" in text:
                return c
            if "dr" in text and ("grade" in text or "level" in text):
                return c
            if "diagnosis" in text or "grade" in text or "level" in text:
                return c

        raise KeyError(f"Cannot find MESSIDOR label column. columns={list(df.columns)}")

    @staticmethod
    def _extract_label(row, label_col):
        value = row[label_col]

        if isinstance(value, str):
            text = value.strip().lower()

            if text in DR_LABEL_TEXT_TO_INT:
                return DR_LABEL_TEXT_TO_INT[text]

            m = re.search(r"([0-4])", text)
            if m:
                return int(m.group(1))

        return label_to_int(value, DR_LABEL_TEXT_TO_INT)

    def _resolve_image_path(self, row, image_col, image_root, basename_to_path, stem_to_path):
        value = str(row[image_col]).strip()
        value = value.replace("\\\\", "/").replace("\\", "/")

        candidates = [
            value,
            os.path.join(self.dataset_dir, value),
            os.path.join(image_root, value),
            os.path.join(image_root, os.path.basename(value)),
            os.path.join(self.dataset_dir, "preprocess", value),
            os.path.join(self.dataset_dir, "preprocess", os.path.basename(value)),
        ]

        suffixes = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        for suffix in suffixes:
            candidates.append(os.path.join(image_root, value + suffix))
            candidates.append(os.path.join(self.dataset_dir, "preprocess", value + suffix))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        basename = os.path.basename(value).lower()
        if basename in basename_to_path:
            return basename_to_path[basename]

        stem = Path(value).stem.lower()
        if stem in stem_to_path:
            return stem_to_path[stem]

        value_text = str(value).strip().lower()
        for known_stem, path in stem_to_path.items():
            if known_stem == value_text or known_stem.endswith(value_text):
                return path

        return None


@DATASET_REGISTRY.register()
class MESIDORDR5(MESSIDORDR5):
    split_filename = "split_zhou_MESIDORDR5.json"


# ---------------------------------------------------------------------------
# MMAC
# ---------------------------------------------------------------------------

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
    dataset_dir_candidates = ["mmac", "MMAC", "MMAC23", "mmac23"]
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
            contains=["label"],
            required=False,
        )
        if label_col is None:
            label_col = find_column(
                df,
                candidates=["class", "grade", "category"],
                contains=["class"],
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
            items.extend(explode_categories(impath, [classname], MMAC_CLASS_TO_LABEL))

        return items
