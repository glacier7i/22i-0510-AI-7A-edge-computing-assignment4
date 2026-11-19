"""Unified inference pipeline: one script for every trained model (teacher, baseline / distilled students, NAS model).

    python inference.py --list                              # available checkpoints
    python inference.py --model mobilenetv2_kd --n 5        # predictions on 5 random test images
    python inference.py --model nas_best --image cat.png    # classify one image file
    python inference.py --model squeezenet_baseline --all   # accuracy on the whole test set
Models are loaded from checkpoints/<name>.pth, the dataset and class names come from the checkpoint itself.
"""
import argparse

import numpy as np
import torch
from PIL import Image

from common import CKPT_DIR, class_names, evaluate, load_data, normalize
from models import load_ckpt


def preprocess(path, dataset):
    """Image file -> normalized 1x3x32x32 tensor."""
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    return normalize(torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 255, dataset)


@torch.no_grad()
def predict(model, x, topk=3):
    """Top-k (probabilities, class indices) for a batch of normalized images."""
    return torch.softmax(model(x), 1).topk(topk, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="checkpoint name without .pth")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--image", help="path of an image to classify")
    ap.add_argument("--n", type=int, default=5, help="number of random test images")
    ap.add_argument("--all", action="store_true", help="evaluate on the full test set")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.list or not args.model:
        print("available models:", ", ".join(sorted(p.stem for p in CKPT_DIR.glob("*.pth"))))
        return
    model, meta = load_ckpt(CKPT_DIR / f"{args.model}.pth", args.device)
    ds, names = meta["dataset"], class_names(meta["dataset"])
    print(f"{args.model}: {meta['arch']} trained on {ds}, test accuracy {meta['test_acc']:.2f}%")

    if args.image:
        probs, idx = predict(model, preprocess(args.image, ds).to(args.device))
        for p, i in zip(probs[0], idx[0]):
            print(f"  {names[i]:<18} {100 * p:5.1f}%")
        return
    _, _, xte, yte = load_data(ds)
    if args.all:
        top1, top5 = evaluate(model, xte, yte, args.device)
        print(f"test top-1 {top1:.2f}%  top-5 {top5:.2f}%")
        return
    pick = torch.randperm(len(xte), generator=torch.Generator().manual_seed(1))[:args.n]
    probs, idx = predict(model, xte[pick].to(args.device), topk=1)
    for k, j in enumerate(pick):
        got, want = names[idx[k, 0]], names[yte[j]]
        print(f"  image {int(j):>5}   true: {want:<15} predicted: {got:<15} ({100 * probs[k, 0]:5.1f}%)  {'ok' if got == want else 'wrong'}")


if __name__ == "__main__":
    main()
