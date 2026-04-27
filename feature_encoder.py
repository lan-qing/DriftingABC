"""
Frozen DINOv2 multi-scale feature encoder used by the drifting loss.
Returns four separate feature maps from intermediate ViT-S/14 transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 encoder.  Returns four feature maps, each of shape
    ``(B, 384, patch_h, patch_h)``, taken at evenly spaced transformer layers
    so that the drifting loss can be computed at multiple scales.
    """

    def __init__(self, model_name: str = "dinov2_vits14", input_size: int = 98):
        super().__init__()
        self.input_size = input_size

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            trust_repo=True,
            force_reload=False,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        num_layers = len(self.model.blocks)
        self.layer_indices = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            3 * num_layers // 4 - 1,
            num_layers - 1,
        ]

        self._features = {}
        for idx in self.layer_indices:
            self.model.blocks[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        self._features = {}
        _ = self.model(x)

        patch_h = self.input_size // 14
        result = []
        for idx in self.layer_indices:
            feat = self._features[idx]
            patch_tokens = feat[:, 1:, :]
            B, N, D = patch_tokens.shape
            feat_map = patch_tokens.permute(0, 2, 1).reshape(B, D, patch_h, patch_h)
            result.append(feat_map)
        return result
