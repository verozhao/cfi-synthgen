import json

import numpy as np
from pycocotools import mask as mask_utils


def mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


class COCODataset:
    def __init__(self, category_names: list[str]):
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._categories = [
            {"id": i + 1, "name": name, "supercategory": "product"}
            for i, name in enumerate(category_names)
        ]
        self._cat_name_to_id = {c["name"]: c["id"] for c in self._categories}
        self._next_image_id = 0
        self._next_ann_id = 0

    def add_image(self, file_name: str, width: int, height: int) -> int:
        image_id = self._next_image_id
        self._next_image_id += 1
        self._images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        return image_id

    def add_annotation(self, image_id: int, category_id: int,
                       modal_mask: np.ndarray, amodal_mask: np.ndarray) -> None:
        ann = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": mask_to_bbox(modal_mask),
            "area": int(modal_mask.sum()),
            "segmentation": mask_to_rle(modal_mask),
            "iscrowd": 0,
            "amodal_segmentation": mask_to_rle(amodal_mask),
        }
        self._next_ann_id += 1
        self._annotations.append(ann)

    def category_id(self, name: str) -> int:
        return self._cat_name_to_id[name]

    def save(self, path: str) -> None:
        data = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": self._categories,
        }
        with open(path, "w") as f:
            json.dump(data, f)
