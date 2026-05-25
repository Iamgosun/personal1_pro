import os

from dassl.data.datasets import DATASET_REGISTRY

from .retina_common import (
    RetinaFundusBase,
    list_images,
    records_to_items,
    split_records_by_primary_label,
)


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
    dataset_dir = "1000x39"
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
            category_dir = os.path.join(self.dataset_dir, category)
            for impath in list_images(category_dir):
                records.append({"impath": impath, "categories": [classname]})

        train_records, val_records, test_records = split_records_by_primary_label(records)
        return (
            records_to_items(train_records, FUNDUS1000X39_CLASS_TO_LABEL),
            records_to_items(val_records, FUNDUS1000X39_CLASS_TO_LABEL),
            records_to_items(test_records, FUNDUS1000X39_CLASS_TO_LABEL),
        )
