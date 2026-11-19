"""Teacher (ResNet-50) and the three student networks, all adapted to 32x32 CIFAR images.

Every model returns logits from forward(x). With forward(x, with_feat=True) it also returns the feature map of an
intermediate stage (8x8 spatial size for all of them), which the feature-based distillation loss uses.
Checkpoint helpers at the bottom let any script rebuild a model from a saved file.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50

ACTS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "silu": nn.SiLU, "hardswish": nn.Hardswish}


def conv_bn_act(cin, cout, k, s=1, act="relu", groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout),
                         ACTS[act]())


# ----------------------------------------------------------------------------- teacher
class ResNet50(nn.Module):
    """torchvision ResNet-50 with a 3x3 stem and no max-pool, so 32x32 inputs are not downsampled too early."""

    feat_ch = 1024

    def __init__(self, num_classes=100):
        super().__init__()
        m = resnet50(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.m = m

    def forward(self, x, with_feat=False):
        m = self.m
        x = m.layer2(m.layer1(m.relu(m.bn1(m.conv1(x)))))
        feat = m.layer3(x)                      # 1024 x 8 x 8
        out = m.fc(m.avgpool(m.layer4(feat)).flatten(1))
        return (out, feat) if with_feat else out


# ----------------------------------------------------------------------------- MobileNetV2 / EfficientNet-Lite0
class MBConv(nn.Module):
    """Inverted residual block: 1x1 expand, depthwise kxk, 1x1 project (no squeeze-excite)."""

    def __init__(self, cin, cout, t, k, s, act):
        super().__init__()
        mid = cin * t
        layers = [conv_bn_act(cin, mid, 1, act=act)] if t != 1 else []
        layers += [conv_bn_act(mid, mid, k, s, act, groups=mid),
                   nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
        self.skip = s == 1 and cin == cout

    def forward(self, x):
        return x + self.block(x) if self.skip else self.block(x)


class MBNet(nn.Module):
    """Generic MobileNet-style network. cfg rows are (expand, channels, repeats, stride, kernel).
    feat_stage: index of the stage whose output is returned as the distillation feature."""

    def __init__(self, cfg, stem, head, act, feat_stage, num_classes):
        super().__init__()
        layers, cin = [conv_bn_act(3, stem, 3, 1, act)], stem
        for si, (t, c, n, s, k) in enumerate(cfg):
            for b in range(n):
                layers.append(MBConv(cin, c, t, k, s if b == 0 else 1, act))
                cin = c
            if si == feat_stage:
                self.feat_idx, self.feat_ch = len(layers) - 1, c
        layers.append(conv_bn_act(cin, head, 1, act=act))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(head, num_classes)

    def forward(self, x, with_feat=False):
        feat = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.feat_idx:
                feat = x
        out = self.fc(x.mean((2, 3)))
        return (out, feat) if with_feat else out


# strides changed from the ImageNet versions so that a 32x32 input ends at a 4x4 map
MOBILENETV2 = [(1, 16, 1, 1, 3), (6, 24, 2, 1, 3), (6, 32, 3, 2, 3), (6, 64, 4, 2, 3),
               (6, 96, 3, 1, 3), (6, 160, 3, 2, 3), (6, 320, 1, 1, 3)]
EFFICIENTNET_LITE0 = [(1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5), (6, 80, 3, 1, 3),
                      (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]


def mobilenetv2(num_classes):
    return MBNet(MOBILENETV2, 32, 1280, "relu6", feat_stage=4, num_classes=num_classes)


def efficientnet_lite0(num_classes):
    # Lite0 = EfficientNet-B0 without squeeze-excite, with ReLU6 and the stem / head left unscaled
    return MBNet(EFFICIENTNET_LITE0, 32, 1280, "relu6", feat_stage=4, num_classes=num_classes)


# ----------------------------------------------------------------------------- SqueezeNet
class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = conv_bn_act(cin, squeeze, 1)
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1, bias=False), nn.BatchNorm2d(e1))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, 1, 1, bias=False), nn.BatchNorm2d(e3))

    def forward(self, x):
        x = self.squeeze(x)
        return torch.relu(torch.cat([self.e1(x), self.e3(x)], 1))


class SqueezeNet(nn.Module):
    """SqueezeNet 1.1 layout (fire modules, 3 max-pools, 1x1 conv classifier) with a 3x3 stride-1 stem and
    batch-norm after every convolution so that it trains from scratch with the same recipe as the others."""

    feat_ch, feat_idx = 256, 6

    def __init__(self, num_classes=100):
        super().__init__()
        pool = lambda: nn.MaxPool2d(3, 2, ceil_mode=True)
        self.features = nn.Sequential(
            conv_bn_act(3, 64, 3), pool(),                                   # 0, 1: stem, then 16x16
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), pool(),             # 2-4: fires at 16x16, then 8x8
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), pool(),        # 5-7: fires at 8x8 (6 = feature), then 4x4
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),                # 8-11: fires at 4x4
            Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        self.classifier = nn.Conv2d(512, num_classes, 1)

    def forward(self, x, with_feat=False):
        feat = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.feat_idx:
                feat = x
        out = self.classifier(x).mean((2, 3))
        return (out, feat) if with_feat else out


BUILDERS = {"resnet50": ResNet50, "mobilenetv2": mobilenetv2, "squeezenet": SqueezeNet,
            "efficientnet_lite0": efficientnet_lite0}
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def build(arch, num_classes=100, config=None):
    """Create a model by name. arch == 'nas' builds the searched network from its config dict."""
    if arch == "nas":
        from nas_space import build_net
        return build_net(config, num_classes)
    return BUILDERS[arch](num_classes)


# ----------------------------------------------------------------------------- checkpoints
def save_ckpt(model, path, arch, dataset, num_classes, config=None, half=False, **info):
    """Checkpoint = weights + everything needed to rebuild the model. half=True stores float16 weights."""
    sd = {k: (v.half() if half and v.is_floating_point() else v).cpu() for k, v in model.state_dict().items()}
    meta = dict(arch=arch, dataset=dataset, num_classes=num_classes, config=config, **info)
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": sd, "meta": meta}, path)


def load_ckpt(path, device="cpu"):
    """Returns (model in eval mode, meta dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = ck["meta"]
    model = build(meta["arch"], meta["num_classes"], meta["config"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return model.to(device).eval(), meta
