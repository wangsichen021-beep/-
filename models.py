from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)
from torchvision.models.resnet import BasicBlock, ResNet


MODEL_NAMES = ("resnet18", "resnet34", "se_resnet18", "se_resnet34")


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        scale = self.pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1, 1)
        return x * scale


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, se_reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.bn2.num_features, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def _replace_fc(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _load_imagenet_into_se_resnet(model: ResNet, arch: str) -> None:
    if arch == "se_resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        baseline = resnet18(weights=weights)
    elif arch == "se_resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1
        baseline = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported SE architecture: {arch}")

    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    unexpected = [key for key in unexpected if key]
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading ImageNet weights: {unexpected}")
    non_se_missing = [key for key in missing if ".se." not in key]
    if non_se_missing:
        raise RuntimeError(f"Missing non-SE keys while loading ImageNet weights: {non_se_missing}")


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if arch not in MODEL_NAMES:
        raise ValueError(f"Unsupported model '{arch}'. Choose from: {', '.join(MODEL_NAMES)}")

    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return _replace_fc(resnet18(weights=weights), num_classes)

    if arch == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return _replace_fc(resnet34(weights=weights), num_classes)

    layers = [2, 2, 2, 2] if arch == "se_resnet18" else [3, 4, 6, 3]
    model = ResNet(SEBasicBlock, layers, num_classes=1000)
    if pretrained:
        _load_imagenet_into_se_resnet(model, arch)
    return _replace_fc(model, num_classes)


def split_parameter_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    high_lr_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc.") or ".se." in name:
            high_lr_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": high_lr_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def describe_parameter_groups(groups: Iterable[dict]) -> list[dict[str, float | int]]:
    description: list[dict[str, float | int]] = []
    for group in groups:
        params = group["params"]
        description.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "num_parameters": sum(param.numel() for param in params),
            }
        )
    return description
