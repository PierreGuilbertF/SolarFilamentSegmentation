import argparse
import csv
import json
import warnings
from time import perf_counter
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from filament_classifier import FEATURE_NAMES, build_mlp, extract_features, read_candidates
from segmentation_utils import annotation_to_rle, load_annotations


def fit_network(network, x, labels, epochs, learning_rate, batch_size, weight_decay):
    import torch

    network.train()
    inputs = torch.tensor(x, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for indices in torch.randperm(len(inputs)).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(inputs[indices]).reshape(-1), targets[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MLP training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        history.append(total_loss / len(inputs))
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs} | BCE: {history[-1]:.6f}", flush=True)
    return history


def train_merge(candidates, records_by_stem, annotations, features, output_dir,
                epochs, learning_rate, batch_size, weight_decay, seed):
    import torch
    from filament_merge import PAIR_FEATURE_NAMES, PairMergeMLP, pair_batches, purity_matches

    started = perf_counter()
    vectors, labels = [], []
    offset = 0
    with (output_dir / "merge_training_features.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filament_id_a", "filament_id_b", "label", *PAIR_FEATURE_NAMES])
        for image_index, (stem, entries) in enumerate(candidates.items(), 1):
            image_features = np.asarray(features[offset:offset + len(entries)])
            offset += len(entries)
            gt = [annotation_to_rle(a) for record in records_by_stem[stem]
                  for a in annotations[record["id"]]]
            matches = purity_matches([rle for _, rle in entries], gt)
            for pairs, batch in pair_batches(entries, image_features):
                for (i, j), vector in zip(pairs, batch):
                    label = int(np.any(matches[i] & matches[j]))
                    vectors.append(vector)
                    labels.append(label)
                    writer.writerow([entries[i][0], entries[j][0], label, *vector])
            print(f"Merge pairs [{image_index}/{len(candidates)}] {stem} | "
                  f"Total pairs: {len(labels)} | Elapsed: {perf_counter() - started:.1f}s", flush=True)
    report = dict(pairs=len(labels), positive=sum(labels), negative=len(labels) - sum(labels),
                  feature_seconds=perf_counter() - started)
    if len(set(labels)) != 2:
        report.update(status="skipped", reason="Merge training requires both positive and negative pairs")
        print(f"WARNING: {report['reason']}; saving filtering model without merging.", flush=True)
        return None, report
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(vectors))
    torch.manual_seed(seed)
    network = PairMergeMLP(len(PAIR_FEATURE_NAMES))
    print("Training pair merge MLP on CPU...", flush=True)
    fit_started = perf_counter()
    history = fit_network(network, x, labels, epochs, learning_rate, batch_size, weight_decay)
    model = dict(version=3, model_type="pair_mlp", feature_names=PAIR_FEATURE_NAMES,
                 mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                 state_dict={key: value.detach().cpu().tolist() for key, value in network.state_dict().items()},
                 purity_threshold=0.7, default_threshold=0.5,
                 training=dict(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
                               weight_decay=weight_decay, seed=seed, optimizer="Adam", loss="BCEWithLogitsLoss"))
    report.update(status="trained", training_loss=history, fit_seconds=perf_counter() - fit_started)
    return model, report


def run(candidates_path, annotations_path, output_dir, regularization_c=0.5,
        model_type="mlp", epochs=300, learning_rate=1e-3, batch_size=256, weight_decay=1e-4, seed=42, merge=True):
    if not np.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("C must be finite and positive")
    if model_type not in ("mlp", "logistic"):
        raise ValueError("model_type must be mlp or logistic")
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    started = perf_counter()
    print("Loading candidates...", flush=True)
    candidates = read_candidates(candidates_path)
    records, annotations = load_annotations(annotations_path)
    records_by_stem = {Path(name).stem: entries for name, entries in records.items()}
    if len(records_by_stem) != len(records):
        raise ValueError("Annotation filenames must have unique stems")
    unknown = set(candidates) - set(records_by_stem)
    if unknown:
        raise ValueError(f"Candidates without annotations: {sorted(unknown)[:5]}")
    features, labels, rows = [], [], []
    for image_index, (stem, entries) in enumerate(candidates.items(), 1):
        gt = [annotation_to_rle(a) for record in records_by_stem[stem]
              for a in annotations[record["id"]]]
        best_ious = (mask_utils.iou([rle for _, rle in entries], gt, [0] * len(gt)).max(axis=1)
                     if gt else np.zeros(len(entries)))
        for (identifier, rle), best_iou in zip(entries, best_ious):
            vector = extract_features(rle)
            label = int(best_iou > 0.5)
            features.append(vector)
            labels.append(label)
            rows.append([identifier, best_iou, label, *vector])
        print(f"[{image_index}/{len(candidates)}] {stem}: {len(entries)} candidates | "
              f"Elapsed: {perf_counter() - started:.1f}s", flush=True)
    feature_seconds = perf_counter() - started
    if len(set(labels)) != 2:
        raise ValueError("Classifier training requires both positive and negative candidates")
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(features))
    fit_started = perf_counter()
    model = {
        "version": 3, "model_type": model_type, "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "positive_class": 1, "label_iou_threshold": 0.5, "default_threshold": 0.5,
    }
    loss_history = []
    if model_type == "mlp":
        import torch

        torch.manual_seed(seed)
        classifier = build_mlp(len(FEATURE_NAMES)).train()
        print("Training filtering MLP on CPU...", flush=True)
        loss_history = fit_network(classifier, x, labels, epochs, learning_rate, batch_size, weight_decay)
        model["state_dict"] = {key: value.detach().cpu().tolist() for key, value in classifier.state_dict().items()}
        model["training"] = dict(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
                                 weight_decay=weight_decay, seed=seed, optimizer="Adam", loss="BCEWithLogitsLoss")
        iterations = epochs
    else:
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs", C=regularization_c)
        print("Fitting L2-regularized logistic regression...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(x, labels)
        model.update(coefficients=classifier.coef_[0].tolist(), intercept=float(classifier.intercept_[0]),
                     regularization={"penalty": "l2", "C": regularization_c})
        iterations = int(classifier.n_iter_[0])
    fit_seconds = perf_counter() - fit_started
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_report = {"status": "disabled"}
    if merge:
        model["merge_model"], merge_report = train_merge(
            candidates, records_by_stem, annotations, features, output_dir,
            epochs, learning_rate, batch_size, weight_decay, seed)
    model_path = output_dir / "filament_classifier.json"
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    with (output_dir / "training_features.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filament_id", "max_iou", "label", *FEATURE_NAMES])
        writer.writerows(rows)
    report = {"merge": merge_report, "feature_seconds": feature_seconds, "fit_seconds": fit_seconds,
              "model_type": model_type, "training_loss": loss_history, "candidates": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
              "images_with_candidates": len(candidates), "iterations": iterations,
              "annotations": str(annotations_path.resolve()), "source_candidates": str(candidates_path.resolve())}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidates: {len(labels)} | Positive: {sum(labels)} | Negative: {len(labels) - sum(labels)}")
    print(f"Model: {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an MLP or logistic classifier on candidate filaments.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C", type=float, default=0.5, help="Logistic regression only: inverse L2 strength; smaller means stronger regularization.")
    parser.add_argument("--model-type", choices=["mlp", "logistic"], default="mlp")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True,
                        help="Train pair merging after filtering (default: enabled).")
    args = parser.parse_args()
    try:
        run(args.candidates, args.annotations, args.output_dir, args.C, args.model_type,
            args.epochs, args.learning_rate, args.batch_size, args.weight_decay, args.seed, args.merge)
    except (ValueError, OSError, KeyError, ConvergenceWarning) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
