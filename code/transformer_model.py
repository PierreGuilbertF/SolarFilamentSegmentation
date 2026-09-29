"""Frozen DINOv3 ConvNeXt encoder and a trainable multiscale decoder."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODEL_TYPE = "dinov3_convnext"
FEATURE_CHANNELS = (96, 192, 384, 768)
PREPROCESS_VERSION = 1


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    dims = config["input_dimensions"]
    if len(dims) != 2 or any(type(x) is not int or x < 32 or x % 32 for x in dims):
        raise ValueError("DINO input_dimensions must be positive multiples of 32")
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Expected model_type={MODEL_TYPE}")
    if config.get("freeze_encoder") is not True or config.get("augment", False):
        raise ValueError("Cached training requires freeze_encoder=true and augment=false")
    if config.get("space_to_depth_stride", 1) != 1:
        raise ValueError("The DINO decoder produces masks at input resolution; use stride=1")
    if config.get("feature_dtype", "float32") not in ("float16", "float32"):
        raise ValueError("feature_dtype must be float16 or float32")
    width = config["decoder_channels"]
    if type(width) is not int or width < 16 or width % 16:
        raise ValueError("decoder_channels must be a multiple of 16")


def feature_spec(config):
    return {"preprocess_version": PREPROCESS_VERSION,
            "encoder_name": config["encoder_name"],
            "encoder_revision": config.get("encoder_revision", "main"),
            "input_dimensions": config["input_dimensions"],
            "feature_dtype": config.get("feature_dtype", "float32"),
            "channels": list(FEATURE_CHANNELS), "strides": [4, 8, 16, 32],
            "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}


def transformers_classes():
    try:
        from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel
    except ImportError as error:
        raise ImportError("Install code/requirements_transformer.txt for DINOv3 support") from error
    return DINOv3ConvNextConfig, DINOv3ConvNextModel


class DinoEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if tuple(backbone.config.hidden_sizes) != FEATURE_CHANNELS:
            raise ValueError("Expected ConvNeXt-Tiny channels: 96, 192, 384, 768")
        self.backbone = backbone.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @classmethod
    def pretrained(cls, config):
        _, model_class = transformers_classes()
        return cls(model_class.from_pretrained(
            config["encoder_name"], revision=config.get("encoder_revision", "main")))

    @classmethod
    def from_snapshot(cls, snapshot):
        config_class, model_class = transformers_classes()
        backbone = model_class(config_class.from_dict(snapshot["backbone_config"]))
        backbone.load_state_dict(snapshot["backbone_state_dict"], strict=True)
        return cls(backbone)

    def snapshot(self):
        return {"backbone_config": self.backbone.config.to_dict(),
                "backbone_state_dict": {k: v.detach().cpu().clone()
                                        for k, v in self.backbone.state_dict().items()}}

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected grayscale images [N, 1, H, W] in [0,1]")
        rgb = (images.float().expand(-1, 3, -1, -1) - self.mean) / self.std
        features = self.backbone(rgb, output_hidden_states=True).hidden_states[-4:]
        for feature, channels, stride in zip(features, FEATURE_CHANNELS, (4, 8, 16, 32)):
            if feature.shape[1:] != (channels, images.shape[-2] // stride, images.shape[-1] // stride):
                raise ValueError("Unexpected DINO feature shape; check Transformers version")
        return tuple(features)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels), nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels), nn.GELU(),
    )


class DinoDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.output_size = tuple(config["input_dimensions"])
        width = config["decoder_channels"]
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in FEATURE_CHANNELS)
        self.blocks = nn.ModuleList(conv_block(width, width) for _ in range(4))
        self.half_resolution = conv_block(width, width // 2)
        self.full_resolution = conv_block(width // 2, width // 2)
        self.head = nn.Conv2d(width // 2, 1, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Decoder requires four feature maps")
        x = self.blocks[3](self.projections[3](features[3]))
        for index in (2, 1, 0):
            skip = self.projections[index](features[index])
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.blocks[index](x + skip)
        x = F.interpolate(x, size=tuple(d // 2 for d in self.output_size), mode="bilinear", align_corners=False)
        x = self.half_resolution(x)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return self.head(self.full_resolution(x))


class DinoSegmentationModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = DinoDecoder(config)
        self.feature_dtype = getattr(torch, config.get("feature_dtype", "float32"))

    def forward(self, images):
        features = self.encoder(images)
        # Match optional cache quantization during inference too.
        features = tuple(feature.to(self.feature_dtype).float() for feature in features)
        return self.decoder(features)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(checkpoint["config"], DinoEncoder.from_snapshot(checkpoint["encoder"]))
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
        return model.eval()
