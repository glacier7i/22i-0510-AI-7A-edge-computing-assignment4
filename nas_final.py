"""Train the best architecture found by nas_search.py from scratch on the full CIFAR-10 training set.
Reads results/nas_best.json  ->  checkpoints/nas_best.pth, results/log_nas_best.csv"""
import argparse
import json

import pandas as pd

from common import CKPT_DIR, RES_DIR, count_params, evaluate, fit, get_device, load_data, seed_all
from models import build, save_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--lr", type=float, default=0.1)
args = ap.parse_args()

best = json.loads((RES_DIR / "nas_best.json").read_text())
cfg = best["config"]
print("architecture:", json.dumps(cfg))

seed_all(0)
dev = get_device()
xtr, ytr, xte, yte = (t.to(dev) for t in load_data("cifar10"))

model, history = build("nas", 10, cfg), []
fit(model, xtr, ytr, args.epochs, args.lr, bs=256, eval_data=(xte, yte), eval_every=5, tag="nas_best", history=history)
top1, top5 = evaluate(model, xte, yte)
print(f"NAS model  test top-1 {top1:.2f}%  top-5 {top5:.2f}%  ({count_params(model) / 1e3:.0f}K params)")
print(f"(proxy-training validation accuracy during the search was {best['val_acc']:.2f}%)")
pd.DataFrame(history).to_csv(RES_DIR / "log_nas_best.csv", index=False)
save_ckpt(model, CKPT_DIR / "nas_best.pth", "nas", "cifar10", 10, config=cfg, test_acc=top1)
