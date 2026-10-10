# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Optional inference-only folding for Samudra's explicit Conv/BatchNorm blocks."""

import torch
from samudra.models.modules.blocks import ConvBlock, ConvNeXtBlock
from torch.nn.utils import fuse_conv_bn_eval


def fold_batch_norm(model: torch.nn.Module) -> list[str]:
    """Fold supported block pairs in place, after checkpoint loading and eval().

    Preserve convolution types and positions: the blocks implement geographic
    padding by inspecting each convolution during their forward method.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("BatchNorm folding requires every module in eval mode")
    folded = []
    for name, block in model.named_modules():
        if isinstance(block, ConvNeXtBlock):
            layers = block.convblock
        elif isinstance(block, ConvBlock):
            layers = block.layers
        else:
            continue
        for index in range(len(layers) - 1):
            conv, norm = layers[index], layers[index + 1]
            if isinstance(conv, torch.nn.Conv2d) and isinstance(
                norm, torch.nn.BatchNorm2d
            ):
                if norm.running_mean is None or norm.running_var is None:
                    raise ValueError(f"Missing BatchNorm running statistics: {name}")
                layers[index] = fuse_conv_bn_eval(conv, norm)
                layers[index + 1] = torch.nn.Identity().eval()
                folded.append(f"{name}:{index}")
    return folded
