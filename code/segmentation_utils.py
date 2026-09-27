import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

IMAGE_SIZE = (2048, 2048)


def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


def annotation_to_rle(annotation):
    segmentation = annotation["segmentation"]
    height, width = IMAGE_SIZE
    if isinstance(segmentation, list):
        if not segmentation:
            return encode_mask(np.zeros(IMAGE_SIZE, dtype=np.uint8))
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if tuple(segmentation["size"]) != IMAGE_SIZE:
        raise ValueError("Annotation RLE dimensions must be 2048 x 2048")
    if isinstance(segmentation["counts"], list):
        return mask_utils.frPyObjects(segmentation, height, width)
    return segmentation


def load_annotations(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records_by_file = defaultdict(list)
    annotations_by_id = defaultdict(list)
    seen_ids = set()
    for record in payload["images"]:
        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate annotation image ID: {record['id']}")
        if Path(record["file_name"]).name != record["file_name"]:
            raise ValueError("Annotation file_name must be a filename without directories")
        if (record["height"], record["width"]) != IMAGE_SIZE:
            raise ValueError("Annotation images must be 2048 x 2048")
        seen_ids.add(record["id"])
        records_by_file[record["file_name"]].append(record)
    for annotation in payload["annotations"]:
        if annotation["image_id"] not in seen_ids:
            raise ValueError(f"Annotation refers to unknown image ID: {annotation['image_id']}")
        if annotation.get("iscrowd", 0):
            raise ValueError("Crowd annotations are not supported by this competition evaluator")
        annotations_by_id[annotation["image_id"]].append(annotation)
    return records_by_file, annotations_by_id


