from pathlib import Path
from dataset_loader import SolarFilamentDatasetLoader
from model import Unet

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def ExportBatch(images: torch.tensor, masks: torch.tensor):
    for k in range(images.shape[0]):
        image = (
            images[k, :, :]
            .detach()
            .permute(1, 2, 0)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        mask = (
            masks[k, 0, :, :]
            .detach()
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        filename = f"/Users/pierre.guilbert/dev/SolarFilaments/output/visualization/{k}_image.png"
        cv2.imwrite(filename, image)
        filename = f"/Users/pierre.guilbert/dev/SolarFilaments/output/visualization/{k}_mask.png"
        cv2.imwrite(filename, mask)


def train_model(config: Path, training_set_payload: Path, output_dir: Path):
    """Train a model using the configuration, training set, and output directory."""

    # Initialize the datasetloader
    dataset_loader = SolarFilamentDatasetLoader(training_set_payload, config)
    train_dataloader = DataLoader(dataset_loader, batch_size=16, shuffle=False)

    # Initialize the model
    model = Unet(config)

    for iteration_idx, (images, masks) in enumerate(train_dataloader):
        print(f"Images: {images.shape}, Masks: {masks.shape}")
        features = model(images)
        print(f"Features: {features.shape}")
        ExportBatch(images, masks)

        break
