from pathlib import Path
import json
import cv2

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode, RandomAffine
from torchvision.transforms import functional as TF


class SolarFilamentDatasetLoader(Dataset):
    def __init__(self, training_set_payload: Path, config_payload: Path, *, augment=True,
                 annotation_sampling="random"):
        if annotation_sampling not in ("random", "all"):
            raise ValueError("annotation_sampling must be random or all")
        self.annotation_sampling = annotation_sampling
        self.augment = augment
        self.geometric_augmentation = RandomAffine(degrees=15, translate=(0.02, 0.02))
        self.contrast_augmentation = ColorJitter(contrast=0.2)
        with config_payload.open(encoding="utf-8") as file:
            config = json.load(file)
        # Dimensions are stored as [height, width].
        self.input_height, self.input_width = config["input_dimensions"]
        stride = config["space_to_depth_stride"]

        if any(
            type(value) is not int or value <= 0
            for value in (self.input_height, self.input_width, stride)
        ):
            raise ValueError(
                "Input dimensions and space_to_depth_stride must be positive integers"
            )

        if self.input_height % stride or self.input_width % stride:
            raise ValueError(
                "Input dimensions must be divisible by space_to_depth_stride"
            )

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
        self.images_name = []
        self.annotation_ids = []
        self.image_ids = [] if annotation_sampling == "all" else None
        self.polygons_by_image = {image["id"]: [] for image in images_payload}
        for annotation in payload["annotations"]:
            self.polygons_by_image[annotation["image_id"]].extend(
                annotation["segmentation"]
            )

        records_by_file = {}
        for record in images_payload:
            records_by_file.setdefault(record["file_name"], []).append(record["id"])
        self.num_images = len(records_by_file)
        if annotation_sampling == "random":
            for filename, ids in records_by_file.items():
                self.images_name.append(str(parent_folder / "train_images" / filename))
                self.annotation_ids.append(ids)
        else:
            for record in images_payload:
                self.images_name.append(str(parent_folder / "train_images" / record["file_name"]))
                self.annotation_ids.append([record["id"]])
                self.image_ids.append(record["id"])
        print(f"Loading: {self.num_images} physical images, {len(images_payload)} annotation sets "
              f"| {len(self)} samples | annotation sampling: {annotation_sampling}")

    def __len__(self):
        return len(self.images_name)

    def _augment(self, image, mask):
        angle, translation, scale, shear = RandomAffine.get_params(
            self.geometric_augmentation.degrees,
            self.geometric_augmentation.translate,
            self.geometric_augmentation.scale,
            self.geometric_augmentation.shear,
            [self.mask_width, self.mask_height],
        )
        # Sample translations on the mask grid, then map them to image pixels.
        image_translation = [
            translation[0] * (self.input_width // self.mask_width),
            translation[1] * (self.input_height // self.mask_height),
        ]
        image = TF.affine(
            self.contrast_augmentation(image),
            angle=angle,
            translate=image_translation,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translation,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        if torch.rand(()).item() < 0.2:
            sigma = torch.empty(()).uniform_(0.3, 0.8).item()
            image = TF.gaussian_blur(image, kernel_size=7, sigma=sigma)
        if torch.rand(()).item() < 0.2:
            sigma = torch.empty(()).uniform_(0.005, 0.015).item()
            image = (image + sigma * torch.randn_like(image)).clamp(0.0, 1.0)
        return image, mask

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
        opencv_image = (
            cv2.resize(
                opencv_image,
                (self.input_width, self.input_height),
                interpolation=interpolation,
            ).astype(np.float32)
            / 255.0
        )
        image = torch.from_numpy(opencv_image).unsqueeze(0)  # [1, H, W]

        ids = self.annotation_ids[idx]
        image_id = ids[torch.randint(len(ids), ()).item()] if len(ids) > 1 else ids[0]
        polygons = self.polygons_by_image[image_id]
        if polygons:
            scale = np.array([self.mask_width / width, self.mask_height / height])
            scaled_polygons = [
                (np.asarray(polygon).reshape(-1, 2) * scale).ravel().tolist()
                for polygon in polygons
            ]
            rles = mask_utils.frPyObjects(
                scaled_polygons, self.mask_height, self.mask_width
            )
            mask = mask_utils.decode(mask_utils.merge(rles))
        else:
            mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)

        mask = torch.from_numpy(mask).to(dtype=torch.float32).unsqueeze(0)
        if self.augment:
            image, mask = self._augment(image, mask)
        return image, mask
