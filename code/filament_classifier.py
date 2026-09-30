import csv
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from collections import defaultdict
from skimage.morphology import skeletonize
from segmentation_utils import IMAGE_SIZE

FEATURE_NAMES = [
    "area", "perimeter", "diameter", "pca_lambda1", "pca_lambda2",
    "pca_ratio", "distance_to_solar_center", "solidity", "extent",
    "eccentricity", "skeleton_length", "area_per_skeleton_length",
]


def read_candidates(path):
    candidates = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Expected filament_id,segmentation_rle CSV columns")
        for row in reader:
            identifier = row["filament_id"]
            stem, separator, index = identifier.rpartition("_")
            if (not separator or not index.isdigit() or not stem
                    or Path(stem).name != stem or stem in {".", ".."}):
                raise ValueError(f"Invalid candidate ID: {identifier}")
            if identifier in seen:
                raise ValueError(f"Duplicate candidate ID: {identifier}")
            seen.add(identifier)
            candidates[stem].append((identifier, {
                "size": list(IMAGE_SIZE), "counts": row["segmentation_rle"].encode("ascii")}))
    return candidates


def extract_features(rle):
    if tuple(rle["size"]) != IMAGE_SIZE:
        raise ValueError("Incorrect candidate mask dimensions")
    mask = mask_utils.decode(rle)
    x, y, width, height = mask_utils.toBbox(rle).astype(int)
    if width <= 0 or height <= 0:
        raise ValueError("Empty candidate")
    mask = np.ascontiguousarray(mask[y:y + height, x:x + width])
    moments = cv2.moments(mask, binaryImage=True)
    area = moments["m00"]
    if not area:
        raise ValueError("Empty candidate")
    center = np.array([moments["m10"], moments["m01"]]) / area
    covariance = np.array([[moments["mu20"], moments["mu11"]],
                           [moments["mu11"], moments["mu02"]]]) / area
    lambda2, lambda1 = np.maximum(np.linalg.eigvalsh(covariance), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    hull = cv2.convexHull(np.concatenate(contours)).reshape(-1, 2).astype(np.float64)
    diameter_squared = 0.0
    for start in range(0, len(hull), 256):
        differences = hull[start:start + 256, None] - hull[None]
        diameter_squared = max(diameter_squared, float(np.square(differences).sum(axis=2).max()))
    # Pixel-square hull keeps area and hull area in the same geometric convention.
    corners = hull[:, None] + np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    hull_area = cv2.contourArea(cv2.convexHull(corners.reshape(-1, 2).astype(np.float32)))
    skeleton_length = int(skeletonize(np.pad(mask.astype(bool), 1), method="zhang").sum())
    solar_center = np.array([(IMAGE_SIZE[1] - 1) / 2, (IMAGE_SIZE[0] - 1) / 2])
    return np.array([
        area, perimeter, np.sqrt(diameter_squared), lambda1, lambda2,
        lambda1 / max(lambda2, 1e-12), np.linalg.norm(center + [x, y] - solar_center),
        area / hull_area, area / (width * height),
        np.sqrt(max(0., 1 - lambda2 / lambda1)) if lambda1 > 0 else 0.,
        skeleton_length, area / max(skeleton_length, 1),
    ], dtype=np.float64)


def build_mlp(n_features):
    from torch import nn

    return nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def load_classifier(path):
    return prepare_classifier(json.loads(path.read_text(encoding="utf-8")))


def prepare_classifier(model):
    model = dict(model)
    names = FEATURE_NAMES
    if model.get("model_type") == "pair_mlp":
        from filament_merge import PAIR_FEATURE_NAMES
        names = PAIR_FEATURE_NAMES
    if model.get("version") not in (2, 3) or model.get("feature_names") != names:
        raise ValueError("Unsupported classifier or feature schema; retrain with geometry-only features")
    kind = model.get("model_type", "logistic")
    if kind not in ("logistic", "mlp", "pair_mlp"):
        raise ValueError("Unsupported classifier type")
    for key in ("mean", "scale") + (("coefficients",) if kind == "logistic" else ()):
        model[key] = np.asarray(model[key], dtype=np.float64)
        if model[key].shape != (len(names),) or not np.isfinite(model[key]).all():
            raise ValueError(f"Invalid classifier {key}")
    if np.any(model["scale"] <= 0):
        raise ValueError("Invalid classifier scale")
    if kind in ("mlp", "pair_mlp"):
        import torch

        if kind == "pair_mlp":
            from filament_merge import PairMergeMLP
            network = PairMergeMLP(len(names))
        else:
            network = build_mlp(len(names))
        state = {key: torch.tensor(value, dtype=torch.float32) for key, value in model["state_dict"].items()}
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Non-finite MLP weights")
        network.load_state_dict(state)
        model["network"] = network.eval()
    elif not np.isfinite(model["intercept"]):
        raise ValueError("Invalid classifier intercept")
    if model.get("merge_model") is not None:
        model["merge_model"] = prepare_classifier(model["merge_model"])
    return model


def predict_probabilities(model, features):
    standardized = (np.asarray(features) - model["mean"]) / model["scale"]
    if model.get("model_type", "logistic") in ("mlp", "pair_mlp"):
        import torch

        model["network"].eval()
        with torch.inference_mode():
            logits = model["network"](torch.as_tensor(standardized, dtype=torch.float32))
            return torch.sigmoid(logits).numpy().reshape(standardized.shape[:-1])
    logits = standardized @ model["coefficients"] + model["intercept"]
    return np.exp(-np.logaddexp(0, -logits))
