import argparse
import csv
import json
from pathlib import Path

import numpy as np

from filament_classifier import extract_features, load_classifier, load_heatmap, predict_probabilities, read_candidates


def run(model_path, candidates_path, heatmaps_dir, output_dir, threshold=0.5):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if (output_dir / "submission.csv").resolve() == candidates_path.resolve():
        raise ValueError("Use a separate output directory to preserve input candidates")
    model = load_classifier(model_path)
    candidates = read_candidates(candidates_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    total, kept = 0, 0
    kept_stems = set()
    with (output_dir / "submission.csv").open("w", newline="", encoding="utf-8") as output, \
            (output_dir / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as scores:
        writer, score_writer = csv.writer(output), csv.writer(scores)
        writer.writerow(["filament_id", "segmentation_rle"])
        score_writer.writerow(["filament_id", "probability", "kept"])
        for stem, entries in candidates.items():
            heatmap = load_heatmap(heatmaps_dir, stem)
            for identifier, rle in entries:
                probability = float(predict_probabilities(model, extract_features(rle, heatmap)))
                keep = probability >= threshold
                total += 1
                kept += int(keep)
                score_writer.writerow([identifier, probability, int(keep)])
                if keep:
                    kept_stems.add(stem)
                    writer.writerow([identifier, rle["counts"].decode("ascii")])
    manifest = candidates_path.parent / "inference_report.json"
    if manifest.exists():
        report = json.loads(manifest.read_text(encoding="utf-8"))
        report["num_predicted_instances"] = kept
        report["images_without_predictions"] = [
            name for name in report["image_files"] if Path(name).stem not in kept_stems]
        report["filament_filter"] = {"model": str(model_path.resolve()), "threshold": threshold}
        (output_dir / "inference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report = {"candidates": total, "kept": kept, "rejected": total - kept, "threshold": threshold,
              "model": str(model_path.resolve()), "source_candidates": str(candidates_path.resolve())}
    (output_dir / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidates: {total} | Kept: {kept} | Rejected: {total - kept}")
    print(f"Submission: {output_dir / 'submission.csv'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Filter candidate filaments with a trained logistic classifier.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--heatmaps-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        run(args.model, args.candidates, args.heatmaps_dir, args.output_dir, args.threshold)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
