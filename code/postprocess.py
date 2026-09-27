import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from model import Unet
from segmentation_utils import IMAGE_SIZE, encode_mask, load_annotations


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return torch.device(name)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "model_state_dict"} <= checkpoint.keys():
        raise ValueError("The checkpoint must contain config and model_state_dict")
    config = checkpoint["config"]
    dimensions = config["input_dimensions"]
    if len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions):
        raise ValueError("input_dimensions must contain positive integers [height, width]")

    with tempfile.TemporaryDirectory() as folder:
        config_path = Path(folder) / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        model = Unet(config_path)

    state = checkpoint["model_state_dict"]
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError("Checkpoint architecture does not match code/model.py and its saved config") from error
    return model.to(device).eval(), config


def heatmap_to_instances(heatmap, threshold):
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Expected a finite two-dimensional heatmap")
    height, width = IMAGE_SIZE
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    binary_mask = (heatmap >= threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    # Stream masks rather than stacking a full-resolution mask for every instance.
    for label in range(1, count):
        yield encode_mask(labels == label)


@torch.inference_mode()
def predict_heatmap(model, image_path, config, device):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if image.shape != IMAGE_SIZE:
        raise ValueError(f"Expected a 2048 x 2048 image, got {image.shape}: {image_path}")
    height, width = config["input_dimensions"]
    interpolation = cv2.INTER_AREA if height <= image.shape[0] and width <= image.shape[1] else cv2.INTER_LINEAR
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)
    prediction = model(image)
    if prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
        raise ValueError(f"Expected model output [1, 1, H, W], got {tuple(prediction.shape)}")
    heatmap = prediction[0, 0].float().cpu().numpy()
    if not np.isfinite(heatmap).all():
        raise ValueError(f"Non-finite model output: {image_path}")
    return cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)


def run(checkpoint_path, images_dir, output_dir, threshold=0.5, annotations_path=None, device_name="auto"):
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not images_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {images_dir}")
    records_by_file = {}
    if annotations_path is not None:
        records_by_file, _ = load_annotations(annotations_path)
        image_paths = [images_dir / name for name in sorted(records_by_file)]
    else:
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not image_paths:
        raise ValueError("No images selected")
    for path in image_paths:
        if not path.is_file():
            raise ValueError(f"Missing annotated image: {path}")
    if len({path.stem for path in image_paths}) != len(image_paths):
        raise ValueError("Image filename stems must be unique for submission IDs")

    device = select_device(device_name)
    model, config = load_model(checkpoint_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    report = {
        "model": str(checkpoint_path.resolve()),
        "config": config,
        "threshold": threshold,
        "connectivity": 8,
        "mask_size": list(IMAGE_SIZE),
        "device": str(device),
        "num_images": len(image_paths),
        "num_predicted_instances": 0,
        "images_without_predictions": [],
        "annotations": str(annotations_path.resolve()) if annotations_path else None,
        "image_files": [path.name for path in image_paths],
        "heatmap_format": "float32 .npy (raw values); .png clipped to [0, 1] for visualization",
    }
    print(f"Device: {device} | Images: {len(image_paths)} | Threshold: {threshold}", flush=True)
    csv_path = output_dir / "submission.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        for index, image_path in enumerate(image_paths, start=1):
            heatmap = predict_heatmap(model, image_path, config, device)
            np.save(heatmap_dir / f"{image_path.stem}.npy", heatmap)
            preview = np.rint(np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            if not cv2.imwrite(str(heatmap_dir / f"{image_path.stem}.png"), preview):
                raise OSError(f"Cannot write heatmap preview for {image_path.name}")
            predictions = list(heatmap_to_instances(heatmap, threshold))
            for instance_id, rle in enumerate(predictions, start=1):
                writer.writerow({
                    "filament_id": f"{image_path.stem}_{instance_id}",
                    "segmentation_rle": rle["counts"].decode("ascii"),
                })
            report["num_predicted_instances"] += len(predictions)
            if not predictions:
                report["images_without_predictions"].append(image_path.name)
            print(f"[{index}/{len(image_paths)}] {image_path.name}: {len(predictions)} filaments", flush=True)

    (output_dir / "inference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Submission: {csv_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run U-Net, save native-resolution heatmaps and export Kaggle filament instances.")
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint with model_state_dict and config.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--annotations", type=Path, help="Optional COCO JSON to select only images listed in this file; no scoring.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    try:
        run(args.model, args.images_dir, args.output_dir, args.threshold, args.annotations, args.device)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
