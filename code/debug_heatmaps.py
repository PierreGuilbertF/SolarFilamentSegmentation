import argparse
import math
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from segmentation_utils import IMAGE_SIZE, annotation_to_rle, load_annotations


def add_legend(image, title, items):
    banner = np.full((90, image.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(banner, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    x = 20
    for label, color in items:
        cv2.rectangle(banner, (x, 50), (x + 20, 70), color, -1)
        cv2.putText(banner, label, (x + 30, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        x += 45 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    return np.concatenate([banner, image], axis=0)


def render(image, heatmap, annotations, threshold, alpha, title):
    background = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    orange, cyan = (0, 140, 255), (255, 255, 0)
    strength = (alpha * np.clip(heatmap, 0, 1))[..., None]
    overlay = np.rint(background * (1 - strength) + np.array(orange) * strength).astype(np.uint8)
    ground_truth = np.zeros(IMAGE_SIZE, dtype=bool)
    for annotation in annotations:
        mask = mask_utils.decode(annotation_to_rle(annotation)).astype(np.uint8)
        ground_truth |= mask.astype(bool)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, cyan, 1, cv2.LINE_8)

    predicted = heatmap >= threshold
    errors = background.copy()
    green, red, blue = (0, 220, 0), (0, 0, 255), (255, 100, 0)
    for region, color in [
        (predicted & ground_truth, green),
        (predicted & ~ground_truth, red),
        (~predicted & ground_truth, blue),
    ]:
        errors[region] = np.rint((1 - alpha) * background[region] + alpha * np.array(color)).astype(np.uint8)
    overlay = add_legend(overlay, title, [
        ("Heatmap: orange opacity proportional to value clipped to [0,1]", orange),
        ("Annotation contours", cyan),
    ])
    errors = add_legend(errors, f"{title} | threshold={threshold:g} | pixel comparison, not instance PQ", [
        ("Prediction AND annotation", green), ("Prediction only", red), ("Annotation only", blue),
    ])
    return overlay, errors


def run(images_dir, heatmaps_dir, annotations_path, output_dir, threshold=0.5, alpha=0.65, image_name=None, limit=None):
    if not math.isfinite(threshold) or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("threshold must be finite and alpha must be in (0, 1]")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    records, annotations = load_annotations(annotations_path)
    names = sorted(records)
    if image_name:
        names = [name for name in names if name == image_name or Path(name).stem == image_name]
        if not names:
            raise ValueError(f"Image not found in annotations: {image_name}")
    if limit:
        names = names[:limit]
    if not names:
        raise ValueError("No images selected")
    for name in names:
        for path in (images_dir / name, heatmaps_dir / f"{Path(name).stem}.npy"):
            if not path.is_file():
                raise ValueError(f"Missing file: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in names:
        image = cv2.imread(str(images_dir / name), cv2.IMREAD_GRAYSCALE)
        heatmap = np.load(heatmaps_dir / f"{Path(name).stem}.npy", allow_pickle=False)
        if image is None or image.shape != IMAGE_SIZE or heatmap.shape != IMAGE_SIZE:
            raise ValueError(f"Expected image and heatmap at 2048 x 2048: {name}")
        if not np.issubdtype(heatmap.dtype, np.number) or np.iscomplexobj(heatmap) or not np.isfinite(heatmap).all():
            raise ValueError(f"Invalid heatmap values: {name}")
        for record in records[name]:
            identifier = str(record['id'])
            title = f"{name} | annotation: {identifier}"
            overlay, errors = render(image, heatmap, annotations[record['id']], threshold, alpha, title)
            for suffix, pixels in [("overlay", overlay), ("errors", errors)]:
                path = output_dir / f"{quote(identifier, safe='')}_{suffix}.png"
                if not cv2.imwrite(str(path), pixels):
                    raise OSError(f"Cannot write {path}")
            count += 1
        print(f"{name}: {len(records[name])} annotation set(s)", flush=True)
    print(f"Exported {2 * count} PNGs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Overlay exported float heatmaps and annotations on solar images.")
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--heatmaps-dir', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for the pixel error view only.")
    parser.add_argument('--alpha', type=float, default=0.65)
    parser.add_argument('--image', help="Optional filename or stem to inspect.")
    parser.add_argument('--limit', type=int, help="Maximum number of physical images to inspect.")
    args = parser.parse_args()
    try:
        run(args.images_dir, args.heatmaps_dir, args.annotations, args.output_dir,
            args.threshold, args.alpha, args.image, args.limit)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
