import os
import pickle
from pathlib import Path


from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class CIFAR_10(DatasetBase):
    dataset_dir = "cifar10"

    classnames = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")

        mkdir_if_missing(self.dataset_dir)
        mkdir_if_missing(self.image_dir)
        mkdir_if_missing(self.split_fewshot_dir)

        train = self._build_split(train=True)
        test = self._build_split(train=False)

        num_shots = cfg.DATASET.NUM_SHOTS

        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-seed_{seed}.pkl",
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train_few = self.generate_fewshot_dataset(
                    train,
                    num_shots=num_shots,
                )
                val = self.generate_fewshot_dataset(
                    train,
                    num_shots=min(num_shots, 4),
                )

                train = train_few
                data = {"train": train, "val": val}

                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            train, val = OxfordPets.split_trainval(train)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train,
            val,
            test,
            subsample=subsample,
        )

        super().__init__(train_x=train, val=val, test=test)

    def _build_split(self, train: bool):
        split_name = "train" if train else "test"
        split_dir = Path(self.image_dir) / split_name

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        items = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for label, classname in enumerate(self.classnames):
            class_dir = split_dir / classname

            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            image_paths = [
                p for p in sorted(class_dir.iterdir())
                if p.is_file() and p.suffix.lower() in valid_exts
            ]

            if len(image_paths) == 0:
                raise RuntimeError(f"No images found in: {class_dir}")

            for impath in image_paths:
                items.append(
                    Datum(
                        impath=str(impath),
                        label=label,
                        classname=classname,
                    )
                )

        return items

