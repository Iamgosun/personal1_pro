import os
from pathlib import Path

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPEG",
    ".JPG",
    ".PNG",
}


def _list_images(root):
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"OOD image root does not exist: {root}")

    images = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in IMG_EXTENSIONS
    ]
    images.sort()

    if len(images) == 0:
        raise RuntimeError(f"No OOD images found under: {root}")

    return images


class OODImageFolderBase(DatasetBase):
    """OOD-only image-folder dataset."""

    dataset_name = None
    candidate_dirs = None

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        image_root = self.resolve_image_root(root)

        images = _list_images(image_root)

        test = [
            Datum(
                impath=str(p),
                label=0,
                classname=self.dataset_name,
            )
            for p in images
        ]

        print(
            f"[OODImageFolder] {self.dataset_name}: "
            f"loaded {len(test)} images from {image_root}"
        )

        super().__init__(train_x=[], val=[], test=test)

    def get_num_classes(self, data_source):
        if len(data_source) == 0:
            return 1
        return super().get_num_classes(data_source)

    def resolve_image_root(self, root):
        candidates = [Path(root) / rel for rel in self.candidate_dirs]

        for path in candidates:
            if path.exists():
                return path

        return candidates[0]



@DATASET_REGISTRY.register()
class TinyImageNetOOD(OODImageFolderBase):
    dataset_name = "TinyImageNet"
    candidate_dirs = [
        "tiny-imagenet-200/test/images",
        "tiny-imagenet-200/test",
        "tinyimagenet/test/images",
        "tinyimagenet/test",
        "tiny-imagenet/test/images",
        "tiny-imagenet/test",
    ]

@DATASET_REGISTRY.register()
class LSUNOOD(OODImageFolderBase):
    dataset_name = "LSUN"
    candidate_dirs = [
        "LSUN",
        "LSUN_resize",
    ]


@DATASET_REGISTRY.register()
class iNaturalistOOD(OODImageFolderBase):
    dataset_name = "iNaturalist"
    candidate_dirs = [
        "iNaturalist/val",
        "inaturalist/val"
    ]