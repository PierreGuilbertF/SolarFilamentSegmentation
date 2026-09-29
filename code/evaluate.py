import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from segmentation_utils import IMAGE_SIZE, annotation_to_rle, load_annotations


EVALUATOR_URL = "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook"


def overlap_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float32)
    return mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T.astype(np.float32)


def score_entry(gt_rles, pred_rles):
    return score_overlaps(overlap_matrix(gt_rles, pred_rles))


def score_overlaps(ious):
    # The official notebook v6 counts all qualifying pairs, without a 1:1 assignment.
    hits = ious > 0.5
    return {
        "tp": int(hits.sum()),
        "fp": int((~hits.any(axis=0)).sum()),
        "fn": int((~hits.any(axis=1)).sum()),
        "matched_iou_sum": float(ious[hits].sum(dtype=np.float64)),
    }


def pq_from_counts(counts):
    denominator = counts["tp"] + 0.5 * counts["fp"] + 0.5 * counts["fn"]
    return counts["matched_iou_sum"] / denominator if denominator else 0.0


def metrics_from_counts(counts):
    tp = counts["tp"]
    denominator = tp + 0.5 * counts["fp"] + 0.5 * counts["fn"]
    return {"pq": pq_from_counts(counts),
            "sq": counts["matched_iou_sum"] / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0}


def load_predictions(path, allowed_stems):
    predictions = defaultdict(list)
    ids = set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Expected CSV columns: filament_id,segmentation_rle")
        for row in reader:
            identifier = row["filament_id"]
            stem, separator, index = identifier.rpartition("_")
            if not separator or not index.isdigit() or stem not in allowed_stems:
                raise ValueError(f"Unknown image or invalid filament_id: {identifier}")
            if identifier in ids:
                raise ValueError(f"Duplicate filament_id: {identifier}")
            ids.add(identifier)
            rle = {"size": list(IMAGE_SIZE), "counts": row["segmentation_rle"].encode("ascii")}
            try:
                decoded = mask_utils.decode(rle)
                if not decoded.any():
                    raise ValueError("Empty instance")
                # Re-encoding also verifies that counts cover exactly the expected grid.
                if mask_utils.encode(np.asfortranarray(decoded))["counts"] != rle["counts"]:
                    raise ValueError("Noncanonical RLE or incorrect dimensions")
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid segmentation_rle for {identifier}: {error}") from error
            predictions[stem].append((identifier, rle))
    return predictions


def distribution(values):
    counts, edges = np.histogram(values, bins=np.linspace(0, 1, 51))
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "histogram_counts": counts.tolist(),
        "histogram_edges": edges.tolist(),
    }


def run(submission_path, annotations_path, output_dir):
    records_by_file, annotations_by_id = load_annotations(annotations_path)
    if not records_by_file:
        raise ValueError("No annotated images to evaluate")
    stems = {Path(name).stem for name in records_by_file}
    if len(stems) != len(records_by_file):
        raise ValueError("Annotation image filename stems must be unique")
    predictions = load_predictions(submission_path, stems)
    manifest_path = submission_path.parent / "inference_report.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = set(records_by_file) - set(manifest["image_files"])
        if missing:
            raise ValueError(f"Inference did not process {len(missing)} annotated images; use the matching annotation subset")

    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0}
    per_image, nonzero_ious, nonzero_dices = [], [], []
    gt_degrees, pred_degrees = Counter(), Counter()
    with (output_dir / "overlap_pairs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "annotation_id", "filament_id", "iou", "dice"])
        for filename, records in sorted(records_by_file.items()):
            entries = predictions[Path(filename).stem]
            pred_rles = [rle for _, rle in entries]
            for record in records:
                annotations = annotations_by_id[record["id"]]
                gt_rles = [annotation_to_rle(a) for a in annotations]
                ious = overlap_matrix(gt_rles, pred_rles)
                counts = score_overlaps(ious)
                for key in totals:
                    totals[key] += counts[key]
                per_image.append({"image_id": record["id"], "file_name": filename,
                                  **counts, **metrics_from_counts(counts)})
                overlaps = ious > 0
                gt_degrees.update(overlaps.sum(axis=1).tolist())
                pred_degrees.update(overlaps.sum(axis=0).tolist())
                for g, p in zip(*np.nonzero(overlaps)):
                    iou = float(ious[g, p])
                    dice = 2 * iou / (1 + iou)
                    nonzero_ious.append(iou)
                    nonzero_dices.append(dice)
                    writer.writerow([record["id"], annotations[g].get("id", int(g)), entries[p][0], iou, dice])

    report = {
        "submission": str(submission_path.resolve()),
        "annotations": str(annotations_path.resolve()),
        "evaluator_reference": EVALUATOR_URL,
        "evaluator_version": 6,
        "matching_iou_threshold": 0.5,
        "num_images": len(records_by_file),
        "num_annotation_images": len(per_image),
        **metrics_from_counts(totals),
        "totals": totals,
        "nonzero_pair_iou": distribution(nonzero_ious),
        "nonzero_pair_dice": distribution(nonzero_dices),
        "predictions_per_gt": dict(sorted(gt_degrees.items())),
        "gt_per_prediction": dict(sorted(pred_degrees.items())),
        "per_annotation_image": per_image,
    }
    (output_dir / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PQ: {report['pq']:.6f} | SQ: {report['sq']:.6f} | RQ: {report['rq']:.6f} | "
          f"TP: {totals['tp']} | FP: {totals['fp']} | FN: {totals['fn']}")
    print(f"Scores: {output_dir / 'scores.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Kaggle filament CSV against COCO annotations.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.submission, args.annotations, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
