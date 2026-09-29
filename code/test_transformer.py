import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from dataset_loader import SolarFilamentDatasetLoader
from postprocess import load_model, run as postprocess
from preprocess_dino import read_image, run as cache_features
from train_transformer import DinoFeatureDataset, train_model
from transformer_model import DinoDecoder, DinoEncoder, DinoSegmentationModel, validate_config


class DinoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cached_training_and_offline_inference(self):
        from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "train_images"
            images_dir.mkdir()
            image = np.zeros((2048, 2048), dtype=np.uint8)
            image[200:1700, 300:1200] = 180
            cv2.imwrite(str(images_dir / "sun.png"), image)
            annotations = root / "annotations.json"
            payload = {
                "images": [{"id": i, "file_name": "sun.png", "height": 2048, "width": 2048} for i in (1, 2)],
                "annotations": [{"id": i, "image_id": i, "segmentation":
                                 [[300, 200, 1200, 200, 1200, 1700 - i * 100, 300, 1700 - i * 100]]}
                                for i in (1, 2)],
            }
            annotations.write_text(json.dumps(payload))
            config = json.loads(Path(__file__).with_name("transformer_configuration.json").read_text())
            config.update(input_dimensions=[32, 64], decoder_channels=16, epochs=1, batch_size=2, num_workers=0)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            # Real HF implementation with reduced depth and random weights; no network access.
            encoder = DinoEncoder(DINOv3ConvNextModel(DINOv3ConvNextConfig(
                hidden_sizes=[96, 192, 384, 768], depths=[1, 1, 1, 1])))
            cache_dir = root / "cache"
            manifest = cache_features(config_path, annotations, images_dir, cache_dir, "cpu", encoder=encoder)
            self.assertEqual(len(manifest["images"]), 1)
            dataset = DinoFeatureDataset(annotations, cache_dir, config)
            self.assertEqual(len(dataset), 2)
            features, mask = dataset[0]
            loader = SolarFilamentDatasetLoader(annotations, config_path, augment=False)
            original_image, original_mask = loader[0]
            torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
            torch.testing.assert_close(original_image, read_image(images_dir / "sun.png", [32, 64]), rtol=0, atol=0)
            live_features = encoder(original_image[None])
            for cached, live in zip(features, live_features):
                torch.testing.assert_close(cached, live[0], rtol=0, atol=0)
            self.assertFalse(torch.equal(dataset[0][1], dataset[1][1]))
            full_model = DinoSegmentationModel(config, encoder).train()
            self.assertFalse(full_model.encoder.training)
            prediction = full_model(original_image[None])
            (prediction - mask[None]).square().mean().backward()
            self.assertTrue(all(p.grad is None for p in encoder.parameters()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in full_model.decoder.parameters()))
            with patch.object(DinoEncoder, "pretrained", side_effect=AssertionError("No encoder download during training")):
                report = train_model(config_path, annotations, cache_dir, root / "training", "cpu")
            self.assertEqual(len(report), 1)
            with patch.object(DINOv3ConvNextModel, "from_pretrained", side_effect=AssertionError("No network for inference")):
                loaded, _ = load_model(root / "training/model.pt", torch.device("cpu"))
                self.assertFalse(loaded.training)
                with torch.no_grad():
                    cached_prediction = loaded.decoder([f[None] for f in features])
                    live_prediction = loaded(original_image[None])
                torch.testing.assert_close(cached_prediction, live_prediction, rtol=1e-5, atol=1e-6)
                result = postprocess(root / "training/model.pt", images_dir, root / "predictions",
                                     annotations_path=annotations, device_name="cpu")
            self.assertEqual(result["num_images"], 1)
            exported = np.load(root / "predictions/heatmaps/sun.npy")
            expected = cv2.resize(live_prediction[0, 0].numpy(), (2048, 2048), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_allclose(exported, expected, rtol=1e-5, atol=1e-6)
            with self.assertRaisesRegex(ValueError, "does not match"):
                DinoFeatureDataset(annotations, cache_dir, {**config, "input_dimensions": [64, 64]})
            with self.assertRaisesRegex(ValueError, "empty cache"):
                cache_features(config_path, annotations, images_dir, cache_dir, "cpu", encoder=encoder)

    def test_config_rejects_incompatible_cached_training(self):
        config = json.loads(Path(__file__).with_name("transformer_configuration.json").read_text())
        for change in ({"augment": True}, {"freeze_encoder": False}, {"input_dimensions": [250, 256]}):
            with self.assertRaises(ValueError):
                validate_config({**config, **change})


if __name__ == "__main__":
    unittest.main()
