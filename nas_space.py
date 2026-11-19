"""NAS search space and architecture generator.

A candidate is a plain dict (JSON friendly):
    {"resolution": 24, "stem": 16,
     "stages": [{"c": 32, "n": 2, "k": 3, "type": "dws", "act": "relu"}, ...]}

resolution : the 32x32 input is resized to this size inside the model, so deployment always feeds 32x32 images
stem       : channels of the first 3x3 convolution
stages     : 2-4 stages (variable depth); each has c channels, n blocks, kernel size k, block type
             ("conv" = standard convolution, "dws" = depthwise separable) and an activation function.
             The first block of every stage except the first uses stride 2.
"""
import torch.nn as nn
import torch.nn.functional as F

from models import conv_bn_act

RESOLUTIONS = [16, 24, 32]
STEM_CHANNELS = [8, 16, 24, 32]
CHANNELS = [8, 16, 24, 32, 48, 64, 96, 128]
BLOCKS_PER_STAGE = [1, 2, 3]
KERNELS = [3, 5, 7]
BLOCK_TYPES = ["conv", "dws"]
ACTIVATIONS = ["relu", "relu6", "silu", "hardswish"]
MAX_STAGES, MIN_STAGES = 4, 2


def suggest_config(trial):
    """Sample one architecture with an Optuna trial."""
    n_stages = trial.suggest_int("n_stages", MIN_STAGES, MAX_STAGES)
    cfg = {"resolution": trial.suggest_categorical("resolution", RESOLUTIONS),
           "stem": trial.suggest_categorical("stem", STEM_CHANNELS), "stages": []}
    for i in range(n_stages):  # parameters of unused stages are never sampled
        cfg["stages"].append({
            "c": trial.suggest_categorical(f"c{i}", CHANNELS),
            "n": trial.suggest_categorical(f"n{i}", BLOCKS_PER_STAGE),
            "k": trial.suggest_categorical(f"k{i}", KERNELS),
            "type": trial.suggest_categorical(f"type{i}", BLOCK_TYPES),
            "act": trial.suggest_categorical(f"act{i}", ACTIVATIONS)})
    return cfg


class Block(nn.Module):
    """One conv block. 'conv': kxk conv. 'dws': depthwise kxk followed by a pointwise 1x1. Residual if shapes match."""

    def __init__(self, cin, cout, k, s, kind, act):
        super().__init__()
        if kind == "conv":
            self.body = conv_bn_act(cin, cout, k, s, act)
        else:
            self.body = nn.Sequential(conv_bn_act(cin, cin, k, s, act, groups=cin), conv_bn_act(cin, cout, 1, act=act))
        self.skip = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.skip else self.body(x)


class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)


def build_net(cfg, num_classes=10):
    """Architecture generator: config dict -> nn.Module."""
    layers = [Resize(cfg["resolution"])] if cfg["resolution"] != 32 else []
    layers.append(conv_bn_act(3, cfg["stem"], 3, 1, cfg["stages"][0]["act"]))
    cin = cfg["stem"]
    for i, st in enumerate(cfg["stages"]):
        for b in range(st["n"]):
            layers.append(Block(cin, st["c"], st["k"], 2 if (i > 0 and b == 0) else 1, st["type"], st["act"]))
            cin = st["c"]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, num_classes))
