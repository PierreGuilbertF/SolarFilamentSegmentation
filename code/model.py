from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_function(activation_name: str, out_channels: int):
    if activation_name == "identity":
        return nn.Identity()
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "relu6":
        return nn.ReLU6()
    elif activation_name == "prelu":
        return nn.PReLU(num_parameters=out_channels)
    elif activation_name == "swish":
        return nn.SiLU()
    elif activation_name == "gelu":
        return nn.GELU()


class ConvWithBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bn_momentum=0.1,
        activation="relu",
    ):
        super().__init__()
        padding = (kernel_size - stride + 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.activation = get_activation_function(activation, out_channels)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class TransposeConvWithBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bn_momentum=0.1,
        activation="relu",
    ):
        super().__init__()
        padding = (kernel_size - stride + 1) // 2
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.activation = get_activation_function(activation, out_channels)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualDWBlock(nn.Module):
    def __init__(self, channels, kernel_size, drop_out=False, activation="relu"):
        super().__init__()
        block = [
            nn.Conv2d(
                channels, channels, kernel_size, groups=channels, padding=1, bias=False
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
        ]
        block.append(nn.BatchNorm2d(channels, momentum=0.1))
        block.append(get_activation_function(activation, channels))

        # drop-out is optional
        if drop_out:
            block.append(nn.Dropout2d(p=0.25))

        block.append(nn.Conv2d(channels, channels, 3, groups=channels, padding=1))
        block.append(get_activation_function(activation, channels))

        self.residual = nn.Sequential(*block)

    def forward(self, x):
        return x + self.residual(x)


class AutoEncoderWithSkips(nn.Module):
    def __init__(self, channels: int, config: Path):
        super().__init__()

        with config.open(encoding="utf-8") as file:
            config_payload = json.load(file)

        self.activation = config_payload["activation_function"]
        self.num_blocks = config_payload["num_blocks"]
        self.num_conv_per_block = config_payload["num_conv_per_block"]
        self.conv_kernel_size = config_payload["conv_kernel_size"]

        down_blocks = []
        downscales = []
        up_blocks = []
        upscales = []

        fusions_1 = []
        fusions_2 = []

        for k in range(self.num_blocks):
            current_channels = 2**k * channels
            next_channels = 2 ** (k + 1) * channels

            # Convolution block for a given resolution during downsampling path
            down_blocks.append(
                nn.Sequential(
                    *[
                        ResidualDWBlock(
                            current_channels,
                            self.conv_kernel_size,
                            activation=self.activation,
                        )
                        for _ in range(self.num_conv_per_block)
                    ]
                )
            )

            # Convolution block for a given resolution during upsampling path
            up_blocks.append(
                nn.Sequential(
                    *[
                        ResidualDWBlock(
                            current_channels,
                            self.conv_kernel_size,
                            activation=self.activation,
                        )
                        for _ in range(self.num_conv_per_block)
                    ]
                )
            )

            # Downsampling convolution
            downscales.append(
                ConvWithBN(
                    current_channels,
                    next_channels,
                    2,
                    stride=2,
                    activation=self.activation,
                )
            )

            # Upsamplnig convolution
            upscales.append(
                TransposeConvWithBN(
                    next_channels,
                    current_channels,
                    4,
                    stride=2,
                    activation=self.activation,
                )
            )

            # Fusion block between downsampling and upsampling blocks
            fusions_1.append(
                nn.Conv2d(current_channels, current_channels, 1, bias=True)
            )
            fusions_2.append(
                nn.Conv2d(current_channels, current_channels, 1, bias=True)
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.downscales = nn.ModuleList(downscales)
        self.upscales = nn.ModuleList(upscales)
        self.fusions_1 = nn.ModuleList(fusions_1)
        self.fusions_2 = nn.ModuleList(fusions_2)
        self.fusion_activation = nn.LeakyReLU(negative_slope=0.025)

        # lowest resolution processing
        self.middle_processing = nn.Sequential(
            *[
                ResidualDWBlock(
                    2**self.num_blocks * channels,
                    self.conv_kernel_size,
                    activation=self.activation,
                )
                for _ in range(self.num_conv_per_block)
            ]
        )

    def forward(self, x):
        # Downscaling
        downscale_features = {}
        for k in range(self.num_blocks):
            y = self.down_blocks[k](x)
            downscale_features[k] = y
            x = self.downscales[k](y)

        # ResConv at lowest resolution
        x = self.middle_processing(x)

        # Upscaling
        for k in range(self.num_blocks):
            reverse_k = self.num_blocks - 1 - k
            x = self.fusion_activation(
                self.fusions_1[reverse_k](self.upscales[reverse_k](x))
                + self.fusions_2[reverse_k](downscale_features[reverse_k])
            )
            x = self.up_blocks[reverse_k](x)

        return x


class Unet(nn.Module):
    def __init__(self, config: Path) -> None:
        super().__init__()

        with config.open(encoding="utf-8") as file:
            config_payload = json.load(file)

        self.activation = config_payload["activation_function"]
        self.space_to_depth_stride = config_payload["space_to_depth_stride"]
        self.input_dim = config_payload["input_dim"]

        # Convert spatial dimension to depth dimension
        self.space_to_depth = ConvWithBN(
            1,
            self.input_dim,
            self.space_to_depth_stride,
            self.space_to_depth_stride,
            activation=self.activation,
        )

        # Core of the unet
        self.auto_encoder = AutoEncoderWithSkips(self.input_dim, config)

        # Decoder head
        self.decoder = torch.nn.Conv2d(self.input_dim, 1, 1, 1, bias=False)
        # Start near zero without blocking gradients through a zero-weight head.
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1e-4)

    def forward(self, x):
        return self.decoder(self.auto_encoder(self.space_to_depth(x)))
