import csv
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from evaluate import load_predictions
from segmentation_utils import IMAGE_SIZE

FEATURE_NAMES = [
    "area", "perimeter", "diameter", "pca_lambda1", "pca_lambda2",
    "pca_ratio", "mean_probability", "median_probability",
    "upper_10_percent_mean_probability", "lower_10_percent_mean_probability",
    "distance_to_solar_center",
]


def read_candidates(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Expected filament_id,segmentation_rle CSV columns")
        stems = {row["filament_id"].rpartition("_")[0] for row in reader}
    if any(not stem or Path(stem).name != stem or stem in {".", ".."} for stem in stems):
        raise ValueError("Invalid candidate image name")
    return load_predictions(path, stems)


def load_heatmap(folder, stem):
    heatmap = np.load(folder / f"{stem}.npy", allow_pickle=False)
    if heatmap.shape != IMAGE_SIZE or not np.isfinite(heatmap).all():
        raise ValueError(f"Expected a finite {IMAGE_SIZE} heatmap for {stem}")
    if np.any((heatmap < 0) | (heatmap > 1)):
        raise ValueError(f"Heatmap must contain probabilities in [0, 1]: {stem}")
    return heatmap


def extract_features(rle, heatmap):
    mask = mask_utils.decode(rle)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty candidate")
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    lambda2, lambda1 = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(points)), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    diameter_squared = 0.0
    for start in range(0, len(hull), 256):
        differences = hull[start:start + 256, None] - hull[None]
        diameter_squared = max(diameter_squared, float(np.square(differences).sum(axis=2).max()))
    probabilities = np.sort(heatmap[ys, xs].astype(np.float64))
    tail_count = max(1, int(np.ceil(0.1 * len(probabilities))))
    solar_center = np.array([(mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2])
    return np.array([
        len(points), perimeter, np.sqrt(diameter_squared), lambda1, lambda2,
        lambda1 / max(lambda2, 1e-12), probabilities.mean(), np.median(probabilities),
        probabilities[-tail_count:].mean(), probabilities[:tail_count].mean(),
        np.linalg.norm(center - solar_center),
    ], dtype=np.float64)


def load_classifier(path):
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("version") != 1 or model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Unsupported classifier or feature schema")
    for key in ("mean", "scale", "coefficients"):
        model[key] = np.asarray(model[key], dtype=np.float64)
        if model[key].shape != (len(FEATURE_NAMES),) or not np.isfinite(model[key]).all():
            raise ValueError(f"Invalid classifier {key}")
    if np.any(model["scale"] <= 0) or not np.isfinite(model["intercept"]):
        raise ValueError("Invalid classifier scale or intercept")
    return model


def predict_probabilities(model, features):
    logits = ((features - model["mean"]) / model["scale"]) @ model["coefficients"] + model["intercept"]
    return np.exp(-np.logaddexp(0, -logits))
