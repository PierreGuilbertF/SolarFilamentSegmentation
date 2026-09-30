import argparse
import csv
import json
from pathlib import Path

import numpy as np

from filament_classifier import extract_features, load_classifier, predict_probabilities, read_candidates


def run(model_path, candidates_path, output_dir, threshold=0.5, merge=True, merge_threshold=0.5):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not np.isfinite(merge_threshold) or not 0 <= merge_threshold <= 1:
        raise ValueError("merge_threshold must be in [0, 1]")
    if (output_dir / "submission.csv").resolve() == candidates_path.resolve():
        raise ValueError("Use a separate output directory to preserve input candidates")
    model = load_classifier(model_path)
    merge_model = model.get("merge_model") if merge else None
    if merge and merge_model is None:
        print("No merge model in checkpoint; applying filtering only.", flush=True)
    candidates = read_candidates(candidates_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    total, kept, output_instances = 0, 0, 0
    kept_stems = set()
    with (output_dir / "submission.csv").open("w", newline="", encoding="utf-8") as output, \
            (output_dir / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as scores, \
            (output_dir / "merge_scores.csv").open("w", newline="", encoding="utf-8") as merge_scores, \
            (output_dir / "merge_membership.csv").open("w", newline="", encoding="utf-8") as membership:
        writer, score_writer = csv.writer(output), csv.writer(scores)
        merge_writer, membership_writer = csv.writer(merge_scores), csv.writer(membership)
        merge_writer.writerow(["filament_id_a", "filament_id_b", "probability", "merge"])
        membership_writer.writerow(["output_filament_id", "source_filament_id"])
        writer.writerow(["filament_id", "segmentation_rle"])
        score_writer.writerow(["filament_id", "probability", "kept"])
        for image_index, (stem, entries) in enumerate(candidates.items(), 1):
            features = np.stack([extract_features(rle) for _, rle in entries])
            probabilities = predict_probabilities(model, features)
            survivors, survivor_features = [], []
            for index, ((identifier, rle), probability) in enumerate(zip(entries, probabilities)):
                probability = float(probability)
                keep = probability >= threshold
                total += 1
                kept += int(keep)
                score_writer.writerow([identifier, probability, int(keep)])
                if keep:
                    kept_stems.add(stem)
                    survivors.append((identifier, rle))
                    survivor_features.append(features[index])
            if merge_model is not None:
                from filament_merge import merge_candidates
                survivors, members = merge_candidates(survivors, survivor_features, merge_model,
                                                       merge_threshold, merge_writer)
            else:
                members = [(identifier, identifier) for identifier, _ in survivors]
            membership_writer.writerows(members)
            output_instances += len(survivors)
            for identifier, rle in survivors:
                writer.writerow([identifier, rle["counts"].decode("ascii")])
            print(f"[{image_index}/{len(candidates)}] {stem}: {len(entries)} candidates", flush=True)
    manifest = candidates_path.parent / "inference_report.json"
    if manifest.exists():
        report = json.loads(manifest.read_text(encoding="utf-8"))
        report["num_predicted_instances"] = output_instances
        report["images_without_predictions"] = [
            name for name in report["image_files"] if Path(name).stem not in kept_stems]
        report["filament_filter"] = {"model": str(model_path.resolve()), "threshold": threshold,
                                     "merge_enabled": merge_model is not None, "merge_threshold": merge_threshold}
        (output_dir / "inference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report = {"output_instances": output_instances, "merged_away": kept - output_instances,
              "merge_enabled": merge_model is not None, "merge_threshold": merge_threshold, "candidates": total, "kept": kept, "rejected": total - kept, "threshold": threshold,
              "model": str(model_path.resolve()), "source_candidates": str(candidates_path.resolve())}
    (output_dir / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidates: {total} | Kept: {kept} | Rejected: {total - kept} | After merging: {output_instances}")
    print(f"Submission: {output_dir / 'submission.csv'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Filter candidate filaments with a trained MLP or logistic classifier.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        run(args.model, args.candidates, args.output_dir, args.threshold, args.merge, args.merge_threshold)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
