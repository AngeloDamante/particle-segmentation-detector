"""Visual Transformer Implementation"""
import torch
import torch.nn as nn
# import torchvision.transforms.functional as F
# import numpy as np
from torchvision.ops import StochasticDepth
from einops import rearrange

#LyaerNorm adapted to code dimension
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

# Overlap Merging, first layer of Encoder to extract features
class OverlapPatchMerging(nn.Sequential):
  def __init__(self, in_channels, out_channels, patch_size, overlap_size):
    super().__init__(
        nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=overlap_size, padding= patch_size//2, bias=False),
        LayerNorm2d(out_channels)
    )

# Self-Attention with reduced complexity
class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        self.reducer = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        LayerNorm2d(channels)
        self.att = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

# Substitute patches position
class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

#  Encoder Block formed with Self-Attention and Mix-FFN, it requires a recursive addiction as in the paper
class ResidualAdd(nn.Module):
    """Just a util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x


class SegFormerEncoderBlock(nn.Sequential):
  def __init__(self, channels, reduction_ratio, numb_heads, mlp_expansion, drop_path_prob):
    super().__init__(
      ResidualAdd(nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, numb_heads),
                ),
      ),
      ResidualAdd(nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                ),
      ),
    )

#   Encoder Stage formed by Encoder Block and Overlap Patches Merging
class SegFormerEncoderStage(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        patch_size,
        overlap_size,
        drop_probs,
        depth,
        reduction_ratio,
        num_heads,
        mlp_expansion,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


# patchify the input
def chunks(data, sizes):
    """
    Given an iterable, returns slices using sizes as indices
    """
    # create patches
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk

# SegFormer Encoder: Encoder Stage iterated along different resolution dimensions
class SegFormerEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# Decoder:takes the F features (BxCxHxW) and return F' features with same spatial and channel size
class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels, widths, scale_factors):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features

# Segmentation Head: extract channels from channels*num_features
class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels, out_channels, num_features):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels)
        )
        self.predict = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

#SegFormer model; at the end we interpolate to have the same size of the input image
class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        widths,
        depths,
        all_num_heads,
        patch_size,
        overlap_sizes,
        reduction_ratios,
        mlp_expansions,
        decoder_channels,
        scale_factors,
        out_channels,
        drop_prob=0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_size,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, out_channels, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return nn.functional.interpolate(input=segmentation, size=x.shape[-2:], mode = "bilinear")