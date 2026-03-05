from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

from junk_qc.models.simple_cnn import SimpleCNN
from junk_qc.models.rotation_invariant_cnn import RotationInvariantCNN


def override_first_conv_in_channels(model: nn.Module, in_ch: int) -> nn.Module:
    """
    Modify the first convolution layer of a torchvision backbone to accept `in_ch`
    channels instead of the default 3.

    Strategy:
      - Keep original weights for the first 3 channels.
      - For additional channels, initialize by averaging the existing weights.

    Args:
        model: Torchvision CNN model (e.g., VGG, ResNet).
        in_ch: Desired number of input channels (e.g., 4).

    Returns:
        The modified model (in-place changes are applied).
    """
    first_conv = None

    if hasattr(model, "features") and isinstance(model.features[0], nn.Conv2d):
        first_conv = model.features[0]
    elif hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        first_conv = model.conv1
    else:
        raise ValueError("Could not locate first Conv2d layer in model.")

    old_weight = first_conv.weight.data
    old_in_ch = old_weight.shape[1]
    if in_ch == old_in_ch:
        return model

    new_weight = torch.zeros(first_conv.out_channels, in_ch, *old_weight.shape[2:])
    new_weight[:, :old_in_ch] = old_weight
    if in_ch > old_in_ch:
        mean_extra = old_weight.mean(dim=1, keepdim=True)
        new_weight[:, old_in_ch:] = mean_extra.repeat(1, in_ch - old_in_ch, 1, 1)

    first_conv.in_channels = in_ch
    first_conv.weight = nn.Parameter(new_weight)
    return model


def unfreeze_last_n_layers(model: nn.Module, n: int) -> None:
    """
    Unfreeze trainable parameters in the last `n` layers (modules) of a model.

    Args:
        model: The neural network model.
        n: Number of leaf modules (with parameters) from the end to unfreeze.

    Notes:
        - All other parameters remain frozen (requires_grad = False).
        - This mirrors QUALIFAI's strategy of unfreezing the last 4 layers of a
          pre-trained backbone.
    """
    for p in model.parameters():
        p.requires_grad = False

    leaf_modules = []
    for m in model.modules():
        params = list(m.parameters(recurse=False))
        if params:
            leaf_modules.append(m)

    for m in leaf_modules[-n:]:
        for p in m.parameters():
            p.requires_grad = True


def build_model(
    arch: str,
    in_ch: int = 4,
    num_classes: int = 2,
    pretrained: bool = False,
    unfreeze_last: int = 0,
) -> nn.Module:
    """
    Factory to construct different model architectures while keeping a uniform API.

    Supported `arch` values:
        - "simple_cnn"
        - "rot_invariant_cnn"
        - "vgg19_bn"
        - "resnet18"

    Args:
        arch: Architecture name.
        in_ch: Number of input channels.
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet-pretrained weights when available.
        unfreeze_last: If > 0 and `pretrained` is True, unfreeze the last N layers.

    Returns:
        nn.Module ready to be trained for junk-vs-rest classification.
    """
    arch = arch.lower()

    if arch == "simple_cnn":
        return SimpleCNN(in_ch=in_ch, num_classes=num_classes)

    if arch == "rot_invariant_cnn":
        return RotationInvariantCNN(in_ch=in_ch, num_classes=num_classes)

    if arch == "vgg19_bn":
        weights = tvm.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vgg19_bn(weights=weights)
        override_first_conv_in_channels(model, in_ch=in_ch)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        override_first_conv_in_channels(model, in_ch=in_ch)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    if pretrained and unfreeze_last > 0:
        unfreeze_last_n_layers(model, n=unfreeze_last)

    return model
