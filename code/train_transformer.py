"""Train only the decoder using precomputed, frozen DINOv3 features."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from postprocess import select_device
from preprocess_dino import sha256
from segmentation_utils import IMAGE_SIZE, load_annotations
from train import create_training_dataloader, normalized_mse_loss
from transformer_model import DinoDecoder, FEATURE_CHANNELS, MODEL_TYPE, feature_spec, load_config


class DinoFeatureDataset(Dataset):
    def __init__(self, annotations_path, cache_dir, config):
        self.cache_dir = cache_dir
        self.config = config
        self.manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["format_version"] != 1 or self.manifest["spec"] != feature_spec(config):
            raise ValueError("Feature cache does not match this encoder/preprocessing/resolution configuration")
        if sha256(cache_dir / "encoder.pt") != self.manifest["encoder_sha256"]:
            raise ValueError("Cached encoder snapshot has changed; rebuild the feature cache")
        records, self.annotations = load_annotations(annotations_path)
        self.records = [record for group in records.values() for record in group]
        if not self.records:
            raise ValueError("No annotation records to train on")
        for name in records:
            entry = self.manifest["images"].get(name)
            if entry is None or not (cache_dir / entry["file"]).is_file():
                raise ValueError(f"Missing cached features: {name}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        entry = self.manifest["images"][record["file_name"]]
        features = torch.load(self.cache_dir / entry["file"], map_location="cpu", weights_only=True)
        height, width = self.config["input_dimensions"]
        if len(features) != 4:
            raise ValueError("Cache must contain four feature maps")
        for feature, channels, stride in zip(features, FEATURE_CHANNELS, (4, 8, 16, 32)):
            if tuple(feature.shape) != (channels, height // stride, width // stride):
                raise ValueError(f"Invalid cached feature shape for {record['file_name']}")
        polygons = []
        scale = np.array([width / IMAGE_SIZE[1], height / IMAGE_SIZE[0]])
        for annotation in self.annotations[record["id"]]:
            if not isinstance(annotation["segmentation"], list):
                raise ValueError("Expected polygon annotations, as in the original training loader")
            polygons.extend((np.asarray(p).reshape(-1, 2) * scale).ravel().tolist()
                            for p in annotation["segmentation"])
        if polygons:
            mask = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygons, height, width)))
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        return [feature.float() for feature in features], torch.from_numpy(mask).float().unsqueeze(0)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_model(config_path, training_set_payload, features_dir, output_dir, device_name="auto"):
    config = load_config(config_path)
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if config.get("scheduler", "none") not in ("none", "cosine"):
        raise ValueError("scheduler must be none or cosine")
    if config.get("foreground_weight", 3.0) <= 0:
        raise ValueError("foreground_weight must be positive")
    torch.manual_seed(config.get("seed", 42))
    device = select_device(device_name)
    dataset = DinoFeatureDataset(training_set_payload, features_dir, config)
    loader = create_training_dataloader(dataset, config["batch_size"], device, config)
    decoder = DinoDecoder(config).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config["learning_rate"])
    scheduler = None
    if config.get("scheduler", "none") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"] * len(loader), eta_min=config.get("min_learning_rate", 0.0))
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    train_decoder = torch.compile(decoder) if config.get("compile", False) else decoder
    encoder_snapshot = torch.load(features_dir / "encoder.pt", map_location="cpu", weights_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Frozen encoder cached | Decoder parameters: "
          f"{sum(p.numel() for p in decoder.parameters()):,}", flush=True)
    report = []
    train_decoder.train()
    for epoch in range(config["epochs"]):
        start = batch_start = perf_counter()
        total_loss, samples, batch_times = 0.0, 0, []
        for features, masks in loader:
            features = [f.to(device, non_blocking=device.type == "cuda") for f in features]
            masks = masks.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                predictions = train_decoder(features)
            loss = normalized_mse_loss(predictions.float(), masks, config.get("foreground_weight", 3.0))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= old_scale:
                scheduler.step()
            total_loss += loss.item() * masks.shape[0]
            samples += masks.shape[0]
            synchronize(device)
            batch_times.append(perf_counter() - batch_start)
            batch_start = perf_counter()
        elapsed = perf_counter() - start
        mean_loss = total_loss / samples
        print(f"Epoch {epoch + 1}/{config['epochs']} | Mean training loss: {mean_loss:.8g} | "
              f"Time: {elapsed:.3f}s", flush=True)
        report.append({"epoch": epoch + 1, "mean_training_loss": mean_loss,
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": elapsed, "batch_elapsed_seconds": batch_times})
        checkpoint = {"model_type": MODEL_TYPE, "config": config, "epoch": epoch + 1,
                      "encoder": encoder_snapshot,
                      "encoder_sha256": dataset.manifest["encoder_sha256"],
                      "decoder_state_dict": {k: v.detach().cpu() for k, v in decoder.state_dict().items()}}
        temporary = output_dir / "model.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output_dir / "model.pt")
        (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-set-payload", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    train_model(args.config, args.training_set_payload, args.features_dir, args.output_dir, args.device)
