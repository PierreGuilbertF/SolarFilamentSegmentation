from pathlib import Path
from dataset_loader import SolarFilamentDatasetLoader
from model import Unet
import json
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def normalized_mse_loss(predicted_masks: torch.Tensor, masks: torch.Tensor):
    return (predicted_masks - masks).square().sum() / masks.sum().clamp_min(1.0)


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

    with config.open(encoding="utf-8") as file:
        config_payload = json.load(file)
    batch_size = config_payload["batch_size"]
    num_epochs = config_payload["epochs"]

    # Initialize the datasetloader
    dataset_loader = SolarFilamentDatasetLoader(training_set_payload, config)
    train_dataloader = DataLoader(dataset_loader, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Unet(config)

    # Initialize the optimizer
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=config_payload.get("learning_rate", 1e-3)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_report = []

    model.train()
    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        total_loss = 0.0
        num_samples = 0
        batch_times = []

        # Include data loading and mask generation in each batch measurement.
        batch_start = perf_counter()
        for iteration_idx, (images, masks) in enumerate(train_dataloader):
            adam_optimizer.zero_grad(set_to_none=True)
            predicted_masks = model(images)
            loss = normalized_mse_loss(predicted_masks, masks)
            loss.backward()
            adam_optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * images.size(0)
            num_samples += images.size(0)
            batch_elapsed = perf_counter() - batch_start
            batch_times.append(batch_elapsed)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Batch {iteration_idx + 1}/{len(train_dataloader)} | "
                f"Loss: {batch_loss:.6f} | Time: {batch_elapsed:.3f}s",
                flush=True,
            )
            batch_start = perf_counter()

        epoch_elapsed = perf_counter() - epoch_start
        mean_loss = total_loss / num_samples
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Mean training loss: {mean_loss:.6f} | "
            f"Time: {epoch_elapsed:.3f}s",
            flush=True,
        )
        training_report.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": mean_loss,
                "elapsed_seconds": epoch_elapsed,
                "batch_elapsed_seconds": batch_times,
            }
        )
        with (output_dir / "training_report.json").open("w", encoding="utf-8") as file:
            json.dump(training_report, file, indent=2)
