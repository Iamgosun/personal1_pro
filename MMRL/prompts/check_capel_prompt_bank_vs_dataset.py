#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_capel_prompt_bank_vs_dataset.py

实际检查：
1. 读取 /root/autodl-tmp/MMRL/prompts/capel_prompt_bank_all.json
2. 按当前 MMRL/datasets/*.py 的真实读取逻辑，从数据集文件/目录抽取真实类别
3. 检查 prompt bank 的类别名、类别数、类别顺序、每类 prompt 数是否与真实数据集一致

运行示例：
python check_capel_prompt_bank_vs_dataset.py \
  --prompt-json /root/autodl-tmp/MMRL/prompts/capel_prompt_bank_all.json \
  --data-root /root/autodl-tmp/data \
  --out /root/autodl-tmp/MMRL/prompts/capel_prompt_bank_class_check_report.json
"""

import argparse
import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple, Callable


CALTECH_IGNORED = {"BACKGROUND_Google", "Faces_easy"}
CALTECH_NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}

EUROSAT_NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return path


def list_dirs(path: Path) -> List[str]:
    require_dir(path)
    return sorted([p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")])


def camel_to_words(s: str) -> str:
    # ApplyEyeMakeup -> Apply Eye Makeup
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s)


def norm_name(s: str) -> str:
    """
    用于判断“同一类别的不同写法”：
    ApplyEyeMakeup / Apply_Eye_Makeup / apply eye makeup 都会归一化成 apply eye makeup
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = camel_to_words(s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def read_split_classnames_if_exists(split_path: Path) -> List[str] | None:
    """
    Dassl/CoOp 风格 split_zhou_*.json:
    {
      "train": [[impath, label, classname], ...],
      "val": ...,
      "test": ...
    }
    当前多个 loader 会优先读取 split 文件，所以如果 split 存在，应以 split 为准。
    """
    if not split_path.is_file():
        return None

    obj = json.loads(split_path.read_text(encoding="utf-8"))
    label_to_name: Dict[int, str] = {}

    for split in ["train", "val", "test"]:
        for item in obj.get(split, []):
            if len(item) < 3:
                continue
            _, label, classname = item[:3]
            label_to_name[int(label)] = str(classname)

    if not label_to_name:
        raise ValueError(f"No classnames found in split file: {split_path}")

    return [label_to_name[i] for i in sorted(label_to_name)]


def read_ucf101(data_root: Path) -> List[str]:
    d = data_root / "ucf101"
    split = read_split_classnames_if_exists(d / "split_zhou_UCF101.json")
    if split is not None:
        return split

    f = require_file(d / "ucfTrainTestlist" / "classInd.txt")
    pairs = []
    for line in f.read_text(encoding="utf-8").splitlines():
        label, action = line.strip().split(" ")
        label = int(label) - 1
        elements = re.findall(r"[A-Z][^A-Z]*", action)
        renamed_action = "_".join(elements)
        pairs.append((label, renamed_action))
    return [c for _, c in sorted(pairs)]


def read_oxford_pets(data_root: Path) -> List[str]:
    d = data_root / "oxford_pets"
    split = read_split_classnames_if_exists(d / "split_zhou_OxfordPets.json")
    if split is not None:
        return split

    label_to_name = {}
    for fname in ["trainval.txt", "test.txt"]:
        f = require_file(d / "annotations" / fname)
        for line in f.read_text(encoding="utf-8").splitlines():
            imname, label, *_ = line.strip().split(" ")
            label = int(label) - 1
            breed = "_".join(imname.split("_")[:-1]).lower()
            label_to_name[label] = breed
    return [label_to_name[i] for i in sorted(label_to_name)]


def read_dtd(data_root: Path) -> List[str]:
    d = data_root / "dtd"
    split = read_split_classnames_if_exists(d / "split_zhou_DescribableTextures.json")
    if split is not None:
        return split
    return list_dirs(d / "images")


def read_eurosat(data_root: Path) -> List[str]:
    d = data_root / "eurosat"
    split = read_split_classnames_if_exists(d / "split_zhou_EuroSAT.json")
    if split is not None:
        return split
    cats = list_dirs(d / "2750")
    return [EUROSAT_NEW_CNAMES.get(c, c) for c in cats]


def read_food101(data_root: Path) -> List[str]:
    d = data_root / "food-101"
    split = read_split_classnames_if_exists(d / "split_zhou_Food101.json")
    if split is not None:
        return split
    return list_dirs(d / "images")


def read_fgvc_aircraft(data_root: Path) -> List[str]:
    d = data_root / "fgvc_aircraft"
    f = require_file(d / "variants.txt")
    return [x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_stanford_cars(data_root: Path) -> List[str]:
    d = data_root / "stanford_cars"
    split = read_split_classnames_if_exists(d / "split_zhou_StanfordCars.json")
    if split is not None:
        return split

    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("StanfordCars 需要 scipy：pip install scipy") from e

    meta_file = require_file(d / "devkit" / "cars_meta.mat")
    meta = loadmat(meta_file)["class_names"][0]

    out = []
    for i in range(len(meta)):
        classname = meta[i][0]
        names = classname.split(" ")
        year = names.pop(-1)
        names.insert(0, year)
        out.append(" ".join(names))
    return out


def read_caltech101(data_root: Path) -> List[str]:
    d = data_root / "caltech-101"
    split = read_split_classnames_if_exists(d / "split_zhou_Caltech101.json")
    if split is not None:
        return split

    cats = list_dirs(d / "101_ObjectCategories")
    cats = [c for c in cats if c not in CALTECH_IGNORED]
    return [CALTECH_NEW_CNAMES.get(c, c) for c in cats]


def read_oxford_flowers(data_root: Path) -> List[str]:
    d = data_root / "oxford_flowers"
    split = read_split_classnames_if_exists(d / "split_zhou_OxfordFlowers.json")
    if split is not None:
        return split

    f = require_file(d / "cat_to_name.json")
    lab2cname = json.loads(f.read_text(encoding="utf-8"))
    return [lab2cname[str(i)] for i in sorted(map(int, lab2cname.keys()))]


def read_sun397(data_root: Path) -> List[str]:
    d = data_root / "sun397"
    split = read_split_classnames_if_exists(d / "split_zhou_SUN397.json")
    if split is not None:
        return split

    f = require_file(d / "ClassName.txt")
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        classname = line.strip()[1:]  # remove leading "/"
        names = classname.split("/")[1:]
        names = names[::-1]
        out.append(" ".join(names))
    return out


def read_imagenet(data_root: Path) -> List[str]:
    d = data_root / "imagenet"
    f = require_file(d / "classnames.txt")
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(" ")
        out.append(" ".join(parts[1:]))
    return out


READERS: Dict[str, Callable[[Path], List[str]]] = {
    "ucf101": read_ucf101,
    "food101": read_food101,
    "dtd": read_dtd,
    "stanfordcars": read_stanford_cars,
    "stanford_cars": read_stanford_cars,
    "caltech101": read_caltech101,
    "oxfordpets": read_oxford_pets,
    "oxford_pets": read_oxford_pets,
    "oxfordflowers": read_oxford_flowers,
    "oxford_flowers": read_oxford_flowers,
    "flowers102": read_oxford_flowers,
    "eurosat": read_eurosat,
    "fgvcaircraft": read_fgvc_aircraft,
    "fgvc_aircraft": read_fgvc_aircraft,
    "aircraft": read_fgvc_aircraft,
    "sun397": read_sun397,
    "imagenet": read_imagenet,
}


def canonical_dataset_key(k: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", k.lower())


def get_expected_prompt_count(root: dict, ds_obj: dict) -> int:
    if "prompts_per_class" in ds_obj:
        return int(ds_obj["prompts_per_class"])

    meta = root.get("metadata", {})
    if "prompts_per_class" in meta:
        return int(meta["prompts_per_class"])

    paper_method = meta.get("paper_method", {})
    if "prompts_per_class" in paper_method:
        return int(paper_method["prompts_per_class"])

    return 50


def compare_dataset(ds_key: str, root: dict, ds_obj: dict, actual: List[str]) -> dict:
    classes = ds_obj.get("classes", [])
    expected_prompt_count = get_expected_prompt_count(root, ds_obj)

    template_raw = [str(c.get("raw_name", "")).strip() for c in classes]
    template_name = [str(c.get("name", "")).strip() for c in classes]
    actual_name = list(actual)

    declared_num = ds_obj.get("num_classes", None)

    raw_norm = [norm_name(x) for x in template_raw]
    name_norm = [norm_name(x) for x in template_name]
    actual_norm = [norm_name(x) for x in actual_name]

    hard_order_mismatches = []
    format_only_mismatches = []

    max_len = max(len(classes), len(actual_name))
    for i in range(max_len):
        raw_i = template_raw[i] if i < len(template_raw) else None
        name_i = template_name[i] if i < len(template_name) else None
        actual_i = actual_name[i] if i < len(actual_name) else None

        raw_n = norm_name(raw_i)
        name_n = norm_name(name_i)
        actual_n = norm_name(actual_i)

        if name_n == actual_n or raw_n == actual_n:
            if raw_i != actual_i and name_i != actual_i:
                format_only_mismatches.append({
                    "index": i,
                    "template_raw_name": raw_i,
                    "template_name": name_i,
                    "actual_classname": actual_i,
                    "reason": "normalized_match_but_strict_string_differs",
                })
        else:
            hard_order_mismatches.append({
                "index": i,
                "template_raw_name": raw_i,
                "template_name": name_i,
                "actual_classname": actual_i,
                "template_raw_norm": raw_n,
                "template_name_norm": name_n,
                "actual_norm": actual_n,
            })

    # set-level: prompt bank 中真正用于生成 prompt 的 name 必须能对应真实类别
    actual_set = set(actual_norm)
    name_set = set(name_norm)

    missing_in_actual_by_name = sorted([x for x in name_set if x and x not in actual_set])
    missing_in_prompt_by_name = sorted([x for x in actual_set if x and x not in name_set])

    # raw_name 作为辅助，避免 UCF101 这类 raw/name 两种写法造成误判
    candidate_set = set(raw_norm) | set(name_norm)
    missing_in_prompt_by_raw_or_name = sorted([x for x in actual_set if x and x not in candidate_set])

    prompt_count_errors = []
    prompt_text_class_warnings = []

    for i, c in enumerate(classes):
        prompts = c.get("prompts", [])
        raw_i = c.get("raw_name", "")
        name_i = c.get("name", "")
        name_n = norm_name(name_i)

        if len(prompts) != expected_prompt_count:
            prompt_count_errors.append({
                "index": i,
                "raw_name": raw_i,
                "name": name_i,
                "prompt_count": len(prompts),
                "expected": expected_prompt_count,
            })

        # 非强制检查：prompt 文本里是否包含该类 display name
        # 对 “a photo of dog” 这类模板应该命中；对某些更自然的描述可能不命中，所以只作为 warning。
        for j, p in enumerate(prompts):
            if name_n and name_n not in norm_name(p):
                prompt_text_class_warnings.append({
                    "index": i,
                    "prompt_index": j,
                    "raw_name": raw_i,
                    "name": name_i,
                    "prompt": p,
                    "reason": "prompt_text_does_not_contain_normalized_class_name",
                })
                break

    return {
        "dataset": ds_key,
        "declared_num_classes_in_json": declared_num,
        "template_class_count": len(classes),
        "actual_class_count": len(actual_name),
        "expected_prompts_per_class": expected_prompt_count,

        "num_classes_ok": (
            (declared_num is None or int(declared_num) == len(actual_name))
            and len(classes) == len(actual_name)
        ),
        "name_set_match_ok": (
            len(missing_in_actual_by_name) == 0
            and len(missing_in_prompt_by_name) == 0
        ),
        "raw_or_name_set_match_ok": (
            len(missing_in_actual_by_name) == 0
            and len(missing_in_prompt_by_raw_or_name) == 0
        ),
        "order_normalized_ok": (
            len(hard_order_mismatches) == 0
            and len(classes) == len(actual_name)
        ),
        "all_prompt_counts_ok": len(prompt_count_errors) == 0,

        "missing_in_actual_by_template_name_normalized": missing_in_actual_by_name,
        "missing_in_prompt_by_template_name_normalized": missing_in_prompt_by_name,
        "missing_in_prompt_by_raw_or_name_normalized": missing_in_prompt_by_raw_or_name,

        "hard_order_mismatches": hard_order_mismatches,
        "format_only_mismatches": format_only_mismatches,
        "prompt_count_errors": prompt_count_errors,
        "prompt_text_class_warnings_first_per_class": prompt_text_class_warnings,

        "first_10_template_classes": [
            {
                "raw_name": template_raw[i],
                "name": template_name[i],
            }
            for i in range(min(10, len(classes)))
        ],
        "first_10_actual_classes": actual_name[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-json",
        default="/root/autodl-tmp/MMRL/prompts/capel_prompt_bank_all.json",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="MMRL 的 cfg.DATASET.ROOT，例如 /root/autodl-tmp/data",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Only check selected datasets, e.g. Caltech101 OxfordPets UCF101",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="保存完整 JSON 检查报告",
    )
    args = parser.parse_args()

    prompt_json = require_file(Path(args.prompt_json))
    data_root = require_dir(Path(args.data_root))

    root = json.loads(prompt_json.read_text(encoding="utf-8"))
    datasets = root.get("datasets", {})
    if not datasets:
        raise ValueError(f"No datasets found in {prompt_json}")

    reports = []
    if args.datasets is not None:
        wanted = {canonical_dataset_key(x) for x in args.datasets}
        datasets = {
            k: v for k, v in datasets.items()
            if canonical_dataset_key(k) in wanted
        }

    for ds_key, ds_obj in datasets.items():
        canon = canonical_dataset_key(ds_key)
        reader = READERS.get(canon)

        if reader is None:
            reports.append({
                "dataset": ds_key,
                "error": f"Unsupported dataset key: {ds_key}",
            })
            continue

        actual = reader(data_root)
        reports.append(compare_dataset(ds_key, root, ds_obj, actual))

    final = {
        "prompt_json": str(prompt_json),
        "data_root": str(data_root),
        "reports": reports,
    }

    print("\n========== CAPEL prompt bank class check ==========")

    has_error = False
    for r in reports:
        print(f"\n[{r.get('dataset')}]")

        if "error" in r:
            has_error = True
            print("  ERROR:", r["error"])
            continue

        fields = [
            "declared_num_classes_in_json",
            "template_class_count",
            "actual_class_count",
            "expected_prompts_per_class",
            "num_classes_ok",
            "name_set_match_ok",
            "raw_or_name_set_match_ok",
            "order_normalized_ok",
            "all_prompt_counts_ok",
        ]

        for f in fields:
            print(f"  {f}: {r[f]}")

        print(f"  hard_order_mismatches: {len(r['hard_order_mismatches'])}")
        print(f"  format_only_mismatches: {len(r['format_only_mismatches'])}")
        print(f"  prompt_count_errors: {len(r['prompt_count_errors'])}")
        print(f"  prompt_text_class_warnings: {len(r['prompt_text_class_warnings_first_per_class'])}")

        if (
            not r["num_classes_ok"]
            or not r["raw_or_name_set_match_ok"]
            or not r["order_normalized_ok"]
            or not r["all_prompt_counts_ok"]
        ):
            has_error = True

        if r["hard_order_mismatches"]:
            print("  HARD mismatches, first 20:")
            for x in r["hard_order_mismatches"][:20]:
                print(
                    f"    #{x['index']}: "
                    f"raw={x['template_raw_name']!r}, "
                    f"name={x['template_name']!r}, "
                    f"actual={x['actual_classname']!r}"
                )

        if r["format_only_mismatches"]:
            print("  Format-only mismatches, first 10:")
            for x in r["format_only_mismatches"][:10]:
                print(
                    f"    #{x['index']}: "
                    f"raw={x['template_raw_name']!r}, "
                    f"name={x['template_name']!r}, "
                    f"actual={x['actual_classname']!r}"
                )

        if r["prompt_count_errors"]:
            print("  Prompt count errors, first 20:")
            for x in r["prompt_count_errors"][:20]:
                print(
                    f"    #{x['index']}: "
                    f"raw={x['raw_name']!r}, "
                    f"name={x['name']!r}, "
                    f"prompt_count={x['prompt_count']}, "
                    f"expected={x['expected']}"
                )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved full report to: {out}")

    if has_error:
        print("\nRESULT: FAIL - 存在类别、顺序、数量或 prompt 数不一致。")
        raise SystemExit(1)

    print("\nRESULT: PASS - prompt bank 类别与数据集真实类别一致。")


if __name__ == "__main__":
    main()