"""Baseline: train a student with plain cross-entropy (no teacher).   ->  checkpoints/<student>_baseline.pth

    python train_baseline.py                          # all three students
    python train_baseline.py --student mobilenetv2    # one of them
"""
import argparse

import pandas as pd

from common import CKPT_DIR, RES_DIR, count_params, evaluate, fit, get_device, load_data, seed_all
from models import STUDENTS, build, save_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--student", choices=STUDENTS + ["all"], default="all")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=0.1)
args = ap.parse_args()

dev = get_device()
xtr, ytr, xte, yte = (t.to(dev) for t in load_data("cifar100"))
RES_DIR.mkdir(exist_ok=True)

for name in STUDENTS if args.student == "all" else [args.student]:
    seed_all(0)
    model, history = build(name, 100), []
    fit(model, xtr, ytr, args.epochs, args.lr, eval_data=(xte, yte), eval_every=2, tag=f"{name}_baseline", history=history, resume=True)
    top1, top5 = evaluate(model, xte, yte)
    print(f"{name} baseline  test top-1 {top1:.2f}%  top-5 {top5:.2f}%  ({count_params(model) / 1e6:.2f}M params)\n")
    pd.DataFrame(history).to_csv(RES_DIR / f"log_{name}_baseline.csv", index=False)
    save_ckpt(model, CKPT_DIR / f"{name}_baseline.pth", name, "cifar100", 100, test_acc=top1)
