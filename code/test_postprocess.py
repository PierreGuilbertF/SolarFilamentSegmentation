import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from model import Unet
from postprocess import (
    IMAGE_SIZE, encode_mask, heatmap_to_instances, load_model,
    run,
)

from evaluate import pq_from_counts, score_entry, run as evaluate


class PostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rle_round_trip_and_connectivity(self):
        mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        mask[10:20, 30:40] = 1
        mask[20, 40] = 1  # Diagonal contact belongs to the same 8-connected instance.
        mask[100:110, 150:160] = 1
        rles = list(heatmap_to_instances(mask.astype(np.float32), 0.5))
        self.assertEqual(len(rles), 2)
        decoded = mask_utils.decode(mask_utils.merge(rles))
        np.testing.assert_array_equal(decoded, mask)
        self.assertEqual(int(mask_utils.area(rles[0])), 101)
        counts = rles[0]["counts"].decode("ascii")
        np.testing.assert_array_equal(
            mask_utils.decode({"size": list(IMAGE_SIZE), "counts": counts}),
            mask_utils.decode(rles[0]),
        )

    def test_threshold_and_resize_without_sigmoid(self):
        heatmap = np.full((8, 8), 0.4, dtype=np.float32)
        self.assertEqual(list(heatmap_to_instances(heatmap, 0.5)), [])
        rles = list(heatmap_to_instances(heatmap, 0.3))
        self.assertEqual(len(rles), 1)
        self.assertEqual(int(mask_utils.area(rles[0])), 2048 * 2048)
        heatmap[0, 0] = np.nan
        with self.assertRaises(ValueError):
            list(heatmap_to_instances(heatmap, 0.5))

    def test_pq_matching_and_empty_cases(self):
        full = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        full[20:40, 20:40] = 1
        left = full.copy()
        left[:, 30:] = 0
        right = full - left
        gt = encode_mask(full)
        halves = [encode_mask(left), encode_mask(right)]
        self.assertEqual(pq_from_counts(score_entry([gt], [gt])), 1.0)
        self.assertEqual(score_entry([gt], []), {"tp": 0, "fp": 0, "fn": 1, "matched_iou_sum": 0.0})
        self.assertEqual(score_entry([], [gt])["fp"], 1)
        self.assertEqual(pq_from_counts(score_entry([], [])), 0.0)
        split = score_entry([gt], halves)
        self.assertEqual((split["tp"], split["fp"], split["fn"]), (0, 2, 1))
        merged = score_entry(halves, [gt])
        self.assertEqual((merged["tp"], merged["fp"], merged["fn"]), (0, 1, 2))
        # The public evaluator counts all qualifying pairs, not a 1-to-1 assignment.
        duplicates = score_entry([gt, gt], [gt])
        self.assertEqual((duplicates["tp"], duplicates["fp"], duplicates["fn"]), (2, 0, 0))
        self.assertEqual(pq_from_counts(duplicates), 1.0)

    def test_checkpoint_export_and_multiple_annotators(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_dir = root / "images"
            image_dir.mkdir()
            cv2.imwrite(str(image_dir / "sun.jpeg"), np.zeros(IMAGE_SIZE, dtype=np.uint8))
            cv2.imwrite(str(image_dir / "extra.jpeg"), np.zeros(IMAGE_SIZE, dtype=np.uint8))
            config = {
                "input_dimensions": [32, 48], "space_to_depth_stride": 2, "input_dim": 2,
                "num_blocks": 0, "num_conv_per_block": 0,
                "conv_kernel_size": 3, "activation_function": "relu",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            model = Unet(config_path)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.space_to_depth.bn.bias.fill_(1.0)
                model.decoder.weight.fill_(0.4)
            checkpoint_path = root / "model.pt"
            state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
            torch.save({"config": config, "model_state_dict": state}, checkpoint_path)
            loaded, saved_config = load_model(checkpoint_path, torch.device("cpu"))
            self.assertFalse(loaded.training)
            self.assertEqual(saved_config, config)
            payload = {
                "images": [
                    {"id": f"{i}-sun", "file_name": "sun.jpeg", "height": 2048, "width": 2048}
                    for i in (1, 2)
                ],
                "annotations": [
                    {"image_id": f"{i}-sun", "segmentation": [[0, 0, 2048, 0, 2048, 2048, 0, 2048]]}
                    for i in (1, 2)
                ],
            }
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(payload))
            report = run(checkpoint_path, image_dir, root / "eval", annotations_path=annotations, device_name="cpu")
            self.assertEqual(report["num_images"], 1)
            self.assertEqual(report["num_predicted_instances"], 1)
            heatmap = np.load(root / "eval/heatmaps/sun.npy")
            self.assertEqual(heatmap.shape, IMAGE_SIZE)
            self.assertEqual(heatmap.dtype, np.float32)
            np.testing.assert_allclose(heatmap, 0.8)
            self.assertEqual(cv2.imread(str(root / "eval/heatmaps/sun.png"), 0).shape, IMAGE_SIZE)
            report = evaluate(root / "eval/submission.csv", annotations, root / "scores")
            self.assertEqual(report["totals"]["tp"], 2)
            self.assertEqual(report["nonzero_pair_dice"]["mean"], 1.0)
            self.assertEqual(report["predictions_per_gt"], {1: 2})
            self.assertEqual(report["pq"], 1.0)
            with (root / "eval/submission.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filament_id"], "sun_1")
            decoded = mask_utils.decode({"size": list(IMAGE_SIZE), "counts": rows[0]["segmentation_rle"]})
            self.assertTrue(decoded.all())
            empty = run(checkpoint_path, image_dir, root / "empty", threshold=0.9, annotations_path=annotations, device_name="cpu")
            self.assertEqual(empty["images_without_predictions"], ["sun.jpeg"])
            empty = evaluate(root / "empty/submission.csv", annotations, root / "empty_scores")
            self.assertEqual(empty["pq"], 0.0)
            self.assertEqual(empty["totals"]["fn"], 2)
            self.assertEqual(empty["predictions_per_gt"], {0: 2})
            unlabelled = run(checkpoint_path, image_dir, root / "test", device_name="cpu")
            self.assertNotIn("pq", unlabelled)
            self.assertEqual(unlabelled["num_images"], 2)


if __name__ == "__main__":
    unittest.main()
