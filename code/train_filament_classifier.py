import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from filament_classifier import FEATURE_NAMES, extract_features, load_heatmap, read_candidates
from segmentation_utils import annotation_to_rle, load_annotations


def run(candidates_path, heatmaps_dir, annotations_path, output_dir):
    candidates = read_candidates(candidates_path)
    records, annotations = load_annotations(annotations_path)
    records_by_stem = {Path(name).stem: entries for name, entries in records.items()}
    if len(records_by_stem) != len(records):
        raise ValueError("Annotation filenames must have unique stems")
    unknown = set(candidates) - set(records_by_stem)
    if unknown:
        raise ValueError(f"Candidates without annotations: {sorted(unknown)[:5]}")
    features, labels, rows = [], [], []
    for stem, entries in candidates.items():
        heatmap = load_heatmap(heatmaps_dir, stem)
        gt = [annotation_to_rle(a) for record in records_by_stem[stem]
              for a in annotations[record["id"]]]
        for identifier, rle in entries:
            best_iou = float(mask_utils.iou([rle], gt, [0] * len(gt)).max()) if gt else 0.0
            vector = extract_features(rle, heatmap)
            label = int(best_iou > 0.5)
            features.append(vector)
            labels.append(label)
            rows.append([identifier, best_iou, label, *vector])
    if len(set(labels)) != 2:
        raise ValueError("Logistic regression requires both positive and negative candidates")
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(features))
    classifier = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(x, labels)
    model = {
        "version": 1, "feature_names": FEATURE_NAMES, "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]), "positive_class": 1,
        "label_iou_threshold": 0.5, "default_threshold": 0.5,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "filament_classifier.json"
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    with (output_dir / "training_features.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filament_id", "max_iou", "label", *FEATURE_NAMES])
        writer.writerows(rows)
    report = {"candidates": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
              "images_with_candidates": len(candidates), "iterations": int(classifier.n_iter_[0]),
              "annotations": str(annotations_path.resolve()), "source_candidates": str(candidates_path.resolve())}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidates: {len(labels)} | Positive: {sum(labels)} | Negative: {len(labels) - sum(labels)}")
    print(f"Model: {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a logistic classifier on candidate filaments.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--heatmaps-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.candidates, args.heatmaps_dir, args.annotations, args.output_dir)
    except (ValueError, OSError, KeyError, ConvergenceWarning) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
