from pathlib import Path
import json
import cv2

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, ElasticTransform, InterpolationMode, RandomAffine
from torchvision.transforms import functional as TF
import torch.nn.functional as F

def low_freq_field(image, scale=32, amplitude_max=0.05):
    # image: [C, H, W]
    C, H, W = image.shape

    h_small = max(2, H // scale)
    w_small = max(2, W // scale)

    field = torch.randn(
        1, 1, h_small, w_small,
        device=image.device,
        dtype=image.dtype,
    )

    field = F.interpolate(
        field,
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )[0]  # [1, H, W]

    # Normalisation
    field = field - field.mean()
    field = field / (field.std() + 1e-6)

    amplitude = torch.empty(
        (),
        device=image.device,
    ).uniform_(0.0, amplitude_max)

    return image * (1.0 + amplitude * field)

class SolarFilamentDatasetLoader(Dataset):
    def __init__(self, training_set_payload: Path, config_payload: Path, *, augment=True,
                 annotation_sampling="random"):
        if annotation_sampling not in ("random", "all"):
            raise ValueError("annotation_sampling must be random or all")
        self.annotation_sampling = annotation_sampling
        self.augment = augment
        self.geometric_augmentation = RandomAffine(degrees=3, translate=(0.02, 0.02), )
        self.photometric_augmentation = ColorJitter(
            brightness=0.15,
            contrast=0.20,
        )
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
        self.elastic_probability = float(config.get("elastic_probability", 0.25))
        self.elastic_alpha = float(config.get("elastic_alpha", 10.0))
        self.elastic_sigma = float(config.get("elastic_sigma", 4.0))
        if not 0 <= self.elastic_probability <= 1:
            raise ValueError("elastic_probability must be between 0 and 1")
        if not np.isfinite(self.elastic_alpha) or self.elastic_alpha < 0:
            raise ValueError("elastic_alpha must be finite and nonnegative")
        if not np.isfinite(self.elastic_sigma) or self.elastic_sigma < 0:
            raise ValueError("elastic_sigma must be finite and nonnegative")
        if self.augment and self.elastic_probability > 0:
            kernel = int(8 * self.elastic_sigma + 1)
            kernel += kernel % 2 == 0
            if kernel // 2 >= min(self.input_height, self.input_width):
                raise ValueError("elastic_sigma is too large for input_dimensions")

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

    def _elastic(self, image, mask):
        displacement = ElasticTransform.get_params(
            [self.elastic_alpha] * 2, [self.elastic_sigma] * 2,
            [self.input_height, self.input_width],
        ).to(device=image.device, dtype=image.dtype)
        mask_displacement = displacement
        if image.shape[-2:] != mask.shape[-2:]:
            # Normalized coordinates describe the same warp at either resolution.
            mask_displacement = torch.nn.functional.interpolate(
                displacement.permute(0, 3, 1, 2), size=mask.shape[-2:],
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        return (
            TF.elastic_transform(image, displacement, interpolation=InterpolationMode.BILINEAR, fill=0),
            TF.elastic_transform(mask, mask_displacement, interpolation=InterpolationMode.NEAREST, fill=0),
        )

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

        # Small rotation / translation to model registration errors
        image = TF.affine(
            image,
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

        # Small elastic deformation to model atmospheric rays bending
        if self.elastic_probability > 0 and torch.rand(()).item() < self.elastic_probability:
            image, mask = self._elastic(image, mask)

        # Brightness + contrast
        image = self.photometric_augmentation(image)
        # Gamma
        gamma = torch.empty(()).uniform_(0.9, 1.1).item()
        image = TF.adjust_gamma(image, gamma=gamma)
        # Low frequency photogrammetric perturbations
        # Model clouds etc
        image = low_freq_field(image, scale=32, amplitude_max=0.05)
        # Gaussian blur, modeling atmospher seeing blur
        sigma = torch.empty(()).uniform_(0.1, 1.2).item()
        image = TF.gaussian_blur(image, kernel_size=7, sigma=sigma)
        # Sensor noise modeled with Poisson Distribution
        peak = 10 ** torch.empty(()).uniform_(4.3, 5.2).item()
        image = torch.poisson(image.clamp(0, 1) * peak) / peak
        # Sensor reading noise
        sigma_read = torch.empty(()).uniform_(0.00005, 0.0005).item()
        image = image + torch.randn_like(image) * sigma_read
        image = image.clamp(0, 1)

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
