"""ImageNet-initialised classification backbones with an explicit feature/head split.

Every model is represented as ``backbone -> pooled features -> linear head``. The
split leaves the original Study 1 computation unchanged (the six legacy models
replace exactly the same final layer) while exposing penultimate features for
parent-recoverability probes and for the gradient-reversal parent adversary.
"""

from __future__ import annotations

from dataclasses import dataclass

LEGACY_ARCHITECTURES = (
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v2",
    "vit_b_16",
)
MODERN_ARCHITECTURES = (
    "convnext_tiny",
    "swin_t",
    "regnet_y_1_6gf",
    "efficientnet_v2_s",
    "maxvit_t",
    "dinov2_vits14",
)
ARCHITECTURES = LEGACY_ARCHITECTURES + MODERN_ARCHITECTURES

# Weight identifiers are recorded in every result so that a later torchvision or
# timm default-weight change cannot silently alter the initialisation.
WEIGHT_NAMES = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "resnet50": "ResNet50_Weights.IMAGENET1K_V2",
    "densenet121": "DenseNet121_Weights.IMAGENET1K_V1",
    "efficientnet_b0": "EfficientNet_B0_Weights.IMAGENET1K_V1",
    "mobilenet_v2": "MobileNet_V2_Weights.IMAGENET1K_V2",
    "vit_b_16": "ViT_B_16_Weights.IMAGENET1K_V1",
    "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
    "swin_t": "Swin_T_Weights.IMAGENET1K_V1",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights.IMAGENET1K_V2",
    "efficientnet_v2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    "maxvit_t": "MaxVit_T_Weights.IMAGENET1K_V1",
    "dinov2_vits14": "timm:vit_small_patch14_dinov2.lvd142m",
}


@dataclass(frozen=True)
class BackboneSpec:
    architecture: str
    feature_dim: int
    weights: str


def _torchvision_backbone(architecture: str, pretrained: bool):
    """Return (module producing pooled features, feature dimension)."""
    from torch import nn
    from torchvision import models

    builders = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "densenet121": models.densenet121,
        "efficientnet_b0": models.efficientnet_b0,
        "mobilenet_v2": models.mobilenet_v2,
        "vit_b_16": models.vit_b_16,
        "convnext_tiny": models.convnext_tiny,
        "swin_t": models.swin_t,
        "regnet_y_1_6gf": models.regnet_y_1_6gf,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "maxvit_t": models.maxvit_t,
    }
    weights = None
    if pretrained:
        enum_name, member = WEIGHT_NAMES[architecture].split(".")
        weights = getattr(getattr(models, enum_name), member)
    model = builders[architecture](weights=weights)
    if architecture.startswith(("resnet", "regnet")):
        dim = model.fc.in_features
        model.fc = nn.Identity()
    elif architecture == "densenet121":
        dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif architecture in {"efficientnet_b0", "mobilenet_v2", "efficientnet_v2_s"}:
        # The dropout that precedes the released head stays in the backbone, so
        # training is identical to replacing only the final Linear layer.
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif architecture == "vit_b_16":
        dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
    elif architecture == "convnext_tiny":
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif architecture == "swin_t":
        dim = model.head.in_features
        model.head = nn.Identity()
    elif architecture == "maxvit_t":
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    else:  # pragma: no cover - guarded by the builders lookup above
        raise ValueError(f"Unknown architecture: {architecture}")
    return model, int(dim)


def _timm_backbone(architecture: str, pretrained: bool):
    import timm

    if architecture != "dinov2_vits14":
        raise ValueError(f"Unknown timm architecture: {architecture}")
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=pretrained,
        num_classes=0,
        img_size=224,
    )
    return model, int(model.num_features)


def build_classifier(architecture: str, n_classes: int, *, pretrained: bool = True):
    """Build ``FeatureClassifier`` for one of :data:`ARCHITECTURES`."""
    from torch import nn

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    if architecture == "dinov2_vits14":
        backbone, dim = _timm_backbone(architecture, pretrained)
    else:
        backbone, dim = _torchvision_backbone(architecture, pretrained)

    class FeatureClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(dim, n_classes)
            self.feature_dim = dim

        def features(self, images):
            return self.backbone(images)

        def forward(self, images):
            return self.head(self.features(images))

    spec = BackboneSpec(architecture, dim, WEIGHT_NAMES[architecture] if pretrained else "none")
    return FeatureClassifier(), spec


_GRADIENT_REVERSAL = None


def gradient_reversal(features, scale: float):
    """Identity in the forward pass; multiply gradients by ``-scale`` backwards."""
    global _GRADIENT_REVERSAL
    if _GRADIENT_REVERSAL is None:
        import torch

        class _GradientReversal(torch.autograd.Function):
            @staticmethod
            def forward(ctx, values, coefficient):
                ctx.coefficient = coefficient
                return values.view_as(values)

            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.coefficient * grad_output, None

        _GRADIENT_REVERSAL = _GradientReversal
    return _GRADIENT_REVERSAL.apply(features, float(scale))


def build_parent_adversary(feature_dim: int, n_parents: int, hidden: int = 512):
    """Two-layer parent-identity head used behind the gradient-reversal layer."""
    from torch import nn

    return nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_parents))


def dann_schedule(progress: float, maximum: float) -> float:
    """Ganin et al. (2016) ramp: ``maximum * (2 / (1 + exp(-10 p)) - 1)``."""
    import math

    progress = min(max(float(progress), 0.0), 1.0)
    return float(maximum) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
