"""Shared utilities: CIFAR loading, GPU augmentation, training loop, evaluation and CPU timing."""
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*align=0.*")

ROOT = Path(__file__).resolve().parent
DATA_DIR, CKPT_DIR, RES_DIR, ONNX_DIR = ROOT / "data", ROOT / "checkpoints", ROOT / "results", ROOT / "onnx"

# per-channel mean / std of the training images
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
DATASETS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed=0):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def normalize(x01, dataset):
    """x01: float tensor in [0, 1], NCHW."""
    mean, std = (torch.tensor(s, device=x01.device).view(1, 3, 1, 1) for s in STATS[dataset])
    return (x01 - mean) / std


def class_names(dataset):
    return DATASETS[dataset](DATA_DIR, train=False, download=True).classes


def load_data(dataset):
    """Returns xtr, ytr, xte, yte (normalized float images, long labels), all on the CPU.
    The dataset is only downloaded when it is not already inside data/."""
    out = []
    for train in (True, False):
        d = DATASETS[dataset](DATA_DIR, train=train, download=True)
        x = normalize(torch.from_numpy(d.data).permute(0, 3, 1, 2).float() / 255, dataset)
        out += [x.contiguous(), torch.tensor(d.targets)]
    return out


def split_val(x, y, n=5000, seed=0):
    """Hold out n training images as a validation set."""
    p = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    return x[p[n:]], y[p[n:]], x[p[:n]], y[p[:n]]


def augment(x):
    """Random crop (4 px reflect padding) and horizontal flip, done per sample on the GPU."""
    n, dev = x.size(0), x.device
    xp = F.pad(x, (4,) * 4, mode="reflect")
    ar = torch.arange(32, device=dev)
    i, j = torch.randint(0, 9, (n,), device=dev), torch.randint(0, 9, (n,), device=dev)
    out = xp[torch.arange(n, device=dev)[:, None, None], :, (i[:, None] + ar)[:, :, None], (j[:, None] + ar)[:, None, :]]
    out = out.permute(0, 3, 1, 2)  # advanced indexing puts the channel axis last
    flip = torch.rand(n, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


# ----------------------------------------------------------------------------- model stats
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, size=32):
    """Multiply-accumulates for one image, counted with torch's flop counter (flops / 2)."""
    from torch.utils.flop_counter import FlopCounterMode
    model = model.eval()
    dev = next(model.parameters()).device
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=dev))
    return fc.get_total_flops() // 2


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, x, y, device=None, bs=500):
    """(top-1 %, top-5 %) of a model on (x, y)."""
    device = device or next(model.parameters()).device
    model.eval()
    k = min(5, int(y.max()) + 1)
    top1 = top5 = 0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs].to(device))
        top = out.topk(k, 1).indices.cpu()
        yb = y[i:i + bs].cpu()
        top1 += (top[:, 0] == yb).sum().item()
        top5 += (top == yb[:, None]).any(1).sum().item()
    return 100 * top1 / len(x), 100 * top5 / len(x)


def fit(model, xtr, ytr, epochs, max_lr, loss_fn=None, extra=(), eval_data=None, eval_every=1, wd=5e-4, bs=128,
        tag="train", history=None, verbose=True, resume=False):
    """SGD + one-cycle schedule on GPU-resident data.
    loss_fn(xb, yb) returns the loss for one batch (default: cross-entropy on model(xb)); extra holds additional
    modules that are trained together with the model (e.g. the feature projector in distillation).
    eval_data is only logged, never used to pick a checkpoint: the last epoch is what gets saved.
    resume=True saves checkpoints/resume_<tag>.pt after every epoch and continues from it if the run was
    interrupted; the file is deleted when training finishes."""
    dev = xtr.device
    model.to(dev).to(memory_format=torch.channels_last)
    params = list(model.parameters()) + [p for m in extra for p in m.parameters()]
    opt = torch.optim.SGD(params, lr=max_lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = len(xtr) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=epochs * steps, pct_start=0.2)
    loss_fn = loss_fn or (lambda xb, yb: F.cross_entropy(model(xb), yb))
    history = [] if history is None else history
    state_path, start = CKPT_DIR / f"resume_{tag}.pt", 0
    if resume and state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=True)
        model.load_state_dict(st["model"])
        [m.load_state_dict(s) for m, s in zip(extra, st["extra"])]
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        history[:], start = st["history"], st["epoch"]
        print(f"[{tag}] resuming after epoch {start}", flush=True)
    for ep in range(start, epochs):
        model.train()
        [m.train() for m in extra]
        perm, loss_sum, t0 = torch.randperm(len(xtr), device=dev), 0.0, time.time()
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(xtr[idx]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                loss = loss_fn(xb, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        row = {"tag": tag, "epoch": ep + 1, "train_loss": round(loss_sum.item() / steps, 4)}
        msg = f"[{tag}] epoch {ep + 1:>3}/{epochs}  loss {row['train_loss']:.3f}"
        if eval_data and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            row["test_acc"] = round(evaluate(model, *eval_data, dev)[0], 2)
            msg += f"  acc {row['test_acc']:.2f}%"
        if verbose:
            print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)
        history.append(row)
        if resume:
            CKPT_DIR.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "extra": [m.state_dict() for m in extra], "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "history": history, "epoch": ep + 1}, str(state_path) + ".tmp")
            Path(str(state_path) + ".tmp").replace(state_path)
    if resume:
        state_path.unlink(missing_ok=True)
    return model


# ----------------------------------------------------------------------------- timing
def median_ms(fn, n=200, warmup=30):
    """Median wall-clock time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def best_of(fn, rounds=3, **kw):
    """Smallest median over a few rounds; this filters out CPU frequency and scheduler noise."""
    return min(median_ms(fn, **kw) for _ in range(rounds))
