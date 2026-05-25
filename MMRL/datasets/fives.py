from pathlib import Path
import os

from dassl.data.datasets import DATASET_REGISTRY

from .retina_common import (
    RetinaFundusBase,
    list_images,
    explode_categories,
    records_to_items,
)
from .oxford_pets import OxfordPets


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
    dataset_dir = "fives"
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
            items.extend(
                explode_categories(
                    impath=impath,
                    categories=[classname],
                    class_to_label=FIVES_CLASS_TO_LABEL,
                )
            )

        return items
