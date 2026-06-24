import os
import pickle
import random
from collections import OrderedDict, defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    """
    ImageNet few-shot split with an independent validation support set.

    Few-shot mode:
        train_x = K samples/class from ImageNet/train
        val     = min(K, 4) samples/class from ImageNet/train, disjoint from train_x
        test    = ImageNet/val

    This avoids the old behavior:
        val = test = ImageNet/val
    """

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train_full = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train_full = self.read_data(classnames, "train")
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train_full, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        train = train_full
        val = []

        num_shots = int(cfg.DATASET.NUM_SHOTS)
        if num_shots >= 1:
            seed = int(cfg.SEED)
            num_val_shots = min(num_shots, 4)

            fewshot_cache = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-val_{num_val_shots}-seed_{seed}.pkl",
            )

            if os.path.exists(fewshot_cache):
                print(f"Loading preprocessed ImageNet few-shot train/val from {fewshot_cache}")
                with open(fewshot_cache, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train, val = self.generate_fewshot_train_val_from_train_split(
                    train_full,
                    num_train_shots=num_shots,
                    num_val_shots=num_val_shots,
                    seed=seed,
                )

                data = {
                    "train": train,
                    "val": val,
                    "meta": {
                        "source": "ImageNet/train",
                        "train_shots_per_class": num_shots,
                        "val_shots_per_class": num_val_shots,
                        "seed": seed,
                        "disjoint_train_val": True,
                        "test_source": "ImageNet/val",
                    },
                }
                print(f"Saving preprocessed ImageNet few-shot train/val to {fewshot_cache}")
                with open(fewshot_cache, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train,
            val,
            test,
            subsample=subsample,
        )

        print(
            "[ImageNetFewShotSplit] "
            f"train_x={len(train)}, val={len(val)}, test={len(test)}, "
            f"num_shots={num_shots}, val_shots={min(num_shots, 4) if num_shots >= 1 else 0}"
        )

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary with <folder name>: <class name>."""
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @staticmethod
    def generate_fewshot_train_val_from_train_split(
        dataset,
        num_train_shots,
        num_val_shots,
        seed,
    ):
        tracker = defaultdict(list)
        for item in dataset:
            tracker[int(item.label)].append(item)

        rng = random.Random(seed)

        train = []
        val = []

        for label in sorted(tracker.keys()):
            items = list(tracker[label])
            rng.shuffle(items)

            required = int(num_train_shots) + int(num_val_shots)
            if len(items) < required:
                raise RuntimeError(
                    "Not enough ImageNet/train samples for few-shot train/val split: "
                    f"label={label}, available={len(items)}, required={required}, "
                    f"train_shots={num_train_shots}, val_shots={num_val_shots}"
                )

            train.extend(items[:num_train_shots])
            val.extend(items[num_train_shots:num_train_shots + num_val_shots])

        return train, val
