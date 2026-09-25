from pathlib import Path
import json
import cv2

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


class SolarFilamentDatasetLoader(Dataset):
    def __init__(self, training_set_payload: Path, config_payload: Path):
        with config_payload.open(encoding="utf-8") as file:
            config = json.load(file)
        # Dimensions are stored as [height, width].
        self.input_height, self.input_width = config["input_dimensions"]
        stride = config["space_to_depth_stride"]

        if any(
            type(value) is not int or value <= 0
            for value in (self.input_height, self.input_width, stride)
        ):
            raise ValueError("Input dimensions and space_to_depth_stride must be positive integers")

        if self.input_height % stride or self.input_width % stride:
            raise ValueError("Input dimensions must be divisible by space_to_depth_stride")

        self.mask_height = self.input_height // stride
        self.mask_width = self.input_width // stride

        parent_folder = training_set_payload.parent
        with training_set_payload.open(encoding="utf-8") as file:
            payload = json.load(file)

        if "images" not in payload.keys():
            raise FileNotFoundError(
                "images field is not part of the training-set-payload keys"
            )

        images_payload = payload["images"]
        self.num_images = len(images_payload)

        print(f"Loading: {self.num_images} images")

        self.images_name = []
        self.image_ids = []
        self.polygons_by_image = {image["id"]: [] for image in images_payload}
        for annotation in payload["annotations"]:
            self.polygons_by_image[annotation["image_id"]].extend(
                annotation["segmentation"]
            )

        for image_info in images_payload:
            image_filename = image_info["file_name"]
            full_image_name = f"{parent_folder}/train_images/{image_filename}"
            self.images_name.append(full_image_name)
            self.image_ids.append(image_info["id"])

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        opencv_image = cv2.imread(self.images_name[idx], cv2.IMREAD_GRAYSCALE)
        if opencv_image is None:
            raise FileNotFoundError(f"Cannot read image: {self.images_name[idx]}")

        height, width = opencv_image.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if self.input_height <= height and self.input_width <= width
            else cv2.INTER_LINEAR
        )
        opencv_image = cv2.resize(
            opencv_image, (self.input_width, self.input_height), interpolation=interpolation
        ).astype(np.float32) / 255.0
        image = torch.from_numpy(opencv_image).unsqueeze(0)  # [1, H, W]

        polygons = self.polygons_by_image[self.image_ids[idx]]
        if polygons:
            scale = np.array([self.mask_width / width, self.mask_height / height])
            scaled_polygons = [
                (np.asarray(polygon).reshape(-1, 2) * scale).ravel().tolist()
                for polygon in polygons
            ]
            rles = mask_utils.frPyObjects(scaled_polygons, self.mask_height, self.mask_width)
            mask = mask_utils.decode(mask_utils.merge(rles))
        else:
            mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)

        mask = torch.from_numpy(mask).to(dtype=torch.float32).unsqueeze(0)
        return image, mask
