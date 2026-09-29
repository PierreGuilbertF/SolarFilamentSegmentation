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

EVAL_EVERY = 75

def normalized_mse_loss(predicted_masks, masks, foreground_weight=3.0):
    weights = 1.0 + (foreground_weight - 1.0) * masks
    errors = (predicted_masks - masks).square()
    return (weights * errors).mean()

def segmentation_loss(logits, masks, pos_weight=None):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        masks,
        pos_weight=pos_weight,
    )

    probabilities = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = (probabilities * masks).sum(dim=dims)
    denominator = (
        probabilities.sum(dim=dims)
        + masks.sum(dim=dims)
    )

    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    ).mean()

    return bce + dice_loss

def initialize_data_worker(worker_id):
    cv2.setNumThreads(1)


def create_training_dataloader(dataset, batch_size, device, config, *, shuffle=True):
    cpu_count = os.cpu_count() or 1
    worker_limit = {"cpu": min(2, cpu_count // 4), "mps": 2, "cuda": 8}[device.type]
    default_workers = min(worker_limit, max(0, cpu_count - 1))
    num_workers = config.get("num_workers", default_workers)
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")

    worker_options = {}
    if num_workers > 0:
        prefetch_factor = config.get("prefetch_factor", 2)
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
        shuffle=shuffle,
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


@torch.no_grad()
def validate_model(model, dataloader, device, gt_by_id, threshold=0.5):
    from evaluate import pq_from_counts, score_entry
    from postprocess import heatmap_to_instances

    was_training = model.training
    model.eval()
    total_loss, num_samples, offset = 0.0, 0, 0
    totals = dict(tp=0, fp=0, fn=0, matched_iou_sum=0.0)
    try:
        for images, masks in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            predictions = model(images).float()
            loss = segmentation_loss(predictions, masks.float())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            total_loss += loss.item() * images.shape[0]
            num_samples += images.shape[0]
            predictions = torch.sigmoid(predictions)
            for heatmap in predictions[:, 0].cpu().numpy():
                image_id = dataloader.dataset.image_ids[offset]
                counts = score_entry(gt_by_id[image_id], list(heatmap_to_instances(heatmap, threshold)))
                for key in totals:
                    totals[key] += counts[key]
                offset += 1
    finally:
        model.train(was_training)
    return {"loss": total_loss / num_samples, "pq": pq_from_counts(totals), "totals": totals}


def train_model(config: Path, training_set_payload: Path, output_dir: Path,
                validation_set_payload: Path | None = None):
    """Train a model using the configuration, training set, and output directory."""

    with config.open(encoding="utf-8") as file:
        config_payload = json.load(file)
    batch_size = config_payload["batch_size"]
    num_epochs = config_payload["epochs"]
    validation_threshold = config_payload.get("validation_threshold", 0.5)
    if not np.isfinite(validation_threshold):
        raise ValueError("validation_threshold must be finite")
    if validation_set_payload is not None:
        train_payload = json.loads(training_set_payload.read_text(encoding="utf-8"))
        validation_payload = json.loads(validation_set_payload.read_text(encoding="utf-8"))
        train_names = {i["file_name"] for i in train_payload["images"]}
        validation_names = {i["file_name"] for i in validation_payload["images"]}
        if not train_names or not validation_names:
            raise ValueError("Training and validation splits must be nonempty")
        if train_names & validation_names:
            raise ValueError("Training and validation share physical images; use train_split.json")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}", flush=True)

    # Initialize the datasetloader
    dataset_loader = SolarFilamentDatasetLoader(training_set_payload, config, augment=True)
    train_dataloader = create_training_dataloader(
        dataset_loader, batch_size, device, config_payload
    )
    validation_dataloader = None
    gt_by_id = {}
    if validation_set_payload is not None:
        from segmentation_utils import annotation_to_rle, load_annotations

        validation_dataset = SolarFilamentDatasetLoader(validation_set_payload, config, augment=False, annotation_sampling="all",)
        validation_dataloader = create_training_dataloader(
            validation_dataset, batch_size, device, config_payload, shuffle=False)
        _, annotations = load_annotations(validation_set_payload)
        gt_by_id = {image_id: [annotation_to_rle(a) for a in annotations[image_id]]
                    for image_id in validation_dataset.image_ids}
    print(
        f"Data loading: {train_dataloader.num_workers} workers | "
        f"Prefetch: {train_dataloader.prefetch_factor} | "
        f"Pinned memory: {train_dataloader.pin_memory}",
        flush=True,
    )

    # Initialize the model
    model = Unet(config).to(device)
    if config_payload.get("compile", True):
        model = torch.compile(model, mode="default")
    raw_model = getattr(model, "_orig_mod", model)

    # Initialize the optimizer
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=config_payload.get("learning_rate", 1e-3)
    )

    # Learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    adam_optimizer,
    #    T_max=num_epochs * len(train_dataloader),
    #    eta_min=config_payload.get("min_learning_rate", 0.0),
    #)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_report = []
    best_validation_loss = float("inf")
    best_validation_pq = -float("inf")

    model.train()
    # model = torch.compile(model)
    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        total_loss = torch.zeros((), device=device)
        num_samples = 0
        batch_times = []

        # Include data loading and mask generation in each batch measurement.
        batch_start = perf_counter()
        for iteration_idx, (images, masks) in enumerate(train_dataloader):
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            adam_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                predicted_masks = model(images)
            loss = segmentation_loss(predicted_masks.float(), masks.float())
            loss.backward()
            learning_rate = adam_optimizer.param_groups[0]["lr"]
            adam_optimizer.step()
            #scheduler.step()

            total_loss += loss.detach() * images.size(0)
            num_samples += images.size(0)




        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        epoch_elapsed = perf_counter() - epoch_start
        mean_loss = (total_loss / num_samples).item()
        validation = None
        validation_elapsed = 0.0
        if epoch % EVAL_EVERY == 0:
            ExportBatch(images, masks, predicted_masks)
        if validation_dataloader is not None and epoch % EVAL_EVERY == 0:
            validation_start = perf_counter()
            validation = validate_model(raw_model, validation_dataloader, device, gt_by_id, validation_threshold)
            validation_elapsed = perf_counter() - validation_start

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Mean training loss: {mean_loss:.6f} | "
            f"Time: {epoch_elapsed:.3f}s"
            + (f" | Val loss: {validation['loss']:.8g} | Val PQ: {validation['pq']:.6f}"
               f" | Val time: {validation_elapsed:.3f}s" if validation else ""),
            flush=True,
        )

        training_report.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": mean_loss,
                "elapsed_seconds": epoch_elapsed,
                "validation": validation,
                "validation_elapsed_seconds": validation_elapsed,
            }
        )

        checkpoint = {
            "model_state_dict": raw_model.state_dict(), "config": config_payload,
            "epoch": epoch + 1, "validation": validation,
            "training_annotations": str(training_set_payload.resolve()),
            "validation_annotations": str(validation_set_payload.resolve()) if validation_set_payload else None,
            "validation_threshold": validation_threshold,
        }

        def save_checkpoint(name):
            temporary = output_dir / f"{name}.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output_dir / name)

        save_checkpoint("model.pt")
        if validation is not None:
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                save_checkpoint("best_loss.pt")
            if validation["pq"] > best_validation_pq:
                best_validation_pq = validation["pq"]
                save_checkpoint("best_pq.pt")

        with (output_dir / "training_report.json").open("w", encoding="utf-8") as file:
            json.dump(training_report, file, indent=2)
