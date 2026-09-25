from pathlib import Path
from dataset_loader import SolarFilamentDatasetLoader
from model import Unet
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def normalized_mse_loss(predicted_masks: torch.Tensor, masks: torch.Tensor):
    return (predicted_masks - masks).square().sum() / masks.sum().clamp_min(1.0)


def initialize_data_worker(worker_id):
    cv2.setNumThreads(1)


def create_training_dataloader(dataset, batch_size, device, config):
    cpu_count = os.cpu_count() or 1
    worker_limit = {"cpu": min(2, cpu_count // 4), "mps": 2, "cuda": 8}[device.type]
    default_workers = min(worker_limit, max(0, cpu_count - 1))
    num_workers = config.get("num_workers", default_workers)
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")

    worker_options = {}
    if num_workers > 0:
        prefetch_factor = config.get("prefetch_factor", 4)
        if type(prefetch_factor) is not int or prefetch_factor < 1:
            raise ValueError("prefetch_factor must be a positive integer")
        worker_options = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": True,
            "worker_init_fn": initialize_data_worker,
            "multiprocessing_context": "spawn",
        }

    print(f"Using: {num_workers} workers for batch generation")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        **worker_options,
    )


def ExportBatch(
    images: torch.tensor, masks: torch.tensor, predicted_masks: torch.tensor
):
    print(f"image shape: {images.shape}")
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
        predicted_mask = (
            predicted_masks[k, 0, :, :]
            .detach()
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        filename = f"/home/humans/pierre.guilbert/dev/SolarFilamentSegmentation/output/visualization/{k}_image.png"
        cv2.imwrite(filename, image)
        filename = f"/home/humans/pierre.guilbert/dev/SolarFilamentSegmentation/output/visualization/{k}_mask.png"
        cv2.imwrite(filename, mask)
        filename = f"/home/humans/pierre.guilbert/dev/SolarFilamentSegmentation/output/visualization/{k}_predicted_mask.png"
        cv2.imwrite(filename, predicted_mask)


def train_model(config: Path, training_set_payload: Path, output_dir: Path):
    """Train a model using the configuration, training set, and output directory."""

    with config.open(encoding="utf-8") as file:
        config_payload = json.load(file)
    batch_size = config_payload["batch_size"]
    num_epochs = config_payload["epochs"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}", flush=True)

    # Initialize the datasetloader
    dataset_loader = SolarFilamentDatasetLoader(training_set_payload, config)
    train_dataloader = create_training_dataloader(
        dataset_loader, batch_size, device, config_payload
    )
    print(
        f"Data loading: {train_dataloader.num_workers} workers | "
        f"Prefetch: {train_dataloader.prefetch_factor} | "
        f"Pinned memory: {train_dataloader.pin_memory}",
        flush=True,
    )

    # Initialize the model
    model = Unet(config).to(device)
    model = torch.compile(model, mode="default")

    # Initialize the optimizer
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=config_payload.get("learning_rate", 1e-3)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_report = []

    model.train()
    # model = torch.compile(model)
    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        total_loss = 0.0
        num_samples = 0
        batch_times = []

        # Include data loading and mask generation in each batch measurement.
        batch_start = perf_counter()
        for iteration_idx, (images, masks) in enumerate(train_dataloader):
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            adam_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_masks = model(images)
            loss = normalized_mse_loss(predicted_masks.float(), masks.float())
            loss.backward()
            adam_optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * images.size(0)
            num_samples += images.size(0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
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

        ExportBatch(images, masks, predicted_masks)

        with (output_dir / "training_report.json").open("w", encoding="utf-8") as file:
            json.dump(training_report, file, indent=2)
