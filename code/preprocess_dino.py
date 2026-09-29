"""Cache frozen DINOv3 features once per physical training image."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch

from postprocess import select_device
from segmentation_utils import IMAGE_SIZE, load_annotations
from transformer_model import DinoEncoder, feature_spec, load_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path, dimensions):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != IMAGE_SIZE:
        raise ValueError(f"Expected a readable 2048 x 2048 image: {path}")
    height, width = dimensions
    interpolation = cv2.INTER_AREA if height <= image.shape[0] and width <= image.shape[1] else cv2.INTER_LINEAR
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image).float().unsqueeze(0) / 255.0


def run(config_path, annotations_path, images_dir, output_dir, device_name="auto", encoder=None):
    config = load_config(config_path)
    records, _ = load_annotations(annotations_path)
    if not records:
        raise ValueError("No images to cache")
    names = sorted(records)
    for name in names:
        if not (images_dir / name).is_file():
            raise FileNotFoundError(images_dir / name)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose an empty cache directory to avoid mixing encoder weights or image resolutions")
    device = select_device(device_name)
    encoder = (encoder if encoder is not None else DinoEncoder.pretrained(config)).to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.snapshot(), output_dir / "encoder.pt")
    spec = feature_spec(config)
    entries = {}
    dtype = getattr(torch, config.get("feature_dtype", "float32"))
    for index, name in enumerate(names, 1):
        image = read_image(images_dir / name, config["input_dimensions"])
        features = encoder(image.unsqueeze(0).to(device))
        features = [feature[0].to(device="cpu", dtype=dtype).contiguous() for feature in features]
        if any(not torch.isfinite(feature).all() for feature in features):
            raise ValueError(f"Non-finite features for {name}; use float32 caching")
        filename = hashlib.sha256(name.encode()).hexdigest() + ".pt"
        torch.save(features, output_dir / filename)
        entries[name] = {"file": filename, "image_sha256": sha256(images_dir / name)}
        print(f"[{index}/{len(names)}] Cached {name}", flush=True)
    manifest = {"format_version": 1, "spec": spec,
                "encoder_sha256": sha256(output_dir / "encoder.pt"),
                "images": entries, "annotations": str(annotations_path.resolve())}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cache ready: {output_dir} ({len(entries)} physical images)")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-set-payload", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    run(args.config, args.training_set_payload, args.images_dir, args.output_dir, args.device)
