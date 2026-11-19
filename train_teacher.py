"""Train the ResNet-50 teacher from scratch on CIFAR-100.   ->  checkpoints/teacher_resnet50.pth"""
import argparse

import pandas as pd

from common import CKPT_DIR, RES_DIR, count_params, evaluate, fit, get_device, load_data, seed_all
from models import ResNet50, save_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--lr", type=float, default=0.1)
args = ap.parse_args()

seed_all(0)
dev = get_device()
xtr, ytr, xte, yte = (t.to(dev) for t in load_data("cifar100"))

model, history = ResNet50(100), []
fit(model, xtr, ytr, args.epochs, args.lr, eval_data=(xte, yte), eval_every=2, tag="teacher", history=history, resume=True)

top1, top5 = evaluate(model, xte, yte)
print(f"teacher ResNet-50  test top-1 {top1:.2f}%  top-5 {top5:.2f}%  ({count_params(model) / 1e6:.1f}M params)")
RES_DIR.mkdir(exist_ok=True)
pd.DataFrame(history).to_csv(RES_DIR / "log_teacher.csv", index=False)
# float16 weights keep the file small enough for GitHub; they are converted back to float32 when loaded
save_ckpt(model, CKPT_DIR / "teacher_resnet50.pth", "resnet50", "cifar100", 100, half=True, test_acc=top1)
