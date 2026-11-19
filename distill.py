"""Knowledge distillation: ResNet-50 teacher -> lightweight students.   ->  checkpoints/<student>_kd.pth

Loss for one batch:
    (1 - alpha) * CE(student, labels)                              hard-label term
  + alpha * T^2 * KL(softmax(teacher / T) || softmax(student / T))  soft-target term, temperature T
  + beta * ||proj(student feature) - teacher feature||^2           feature term (8x8 intermediate maps)
The feature term is divided by the variance of the teacher feature so beta does not depend on its scale.
Set --alpha 0 to drop the soft term, --beta 0 to drop the feature term.

    python distill.py                          # all three students
    python distill.py --student squeezenet --beta 0 --tag soft_only
"""
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT_DIR, RES_DIR, count_params, evaluate, fit, get_device, load_data, seed_all
from models import STUDENTS, build, load_ckpt, save_ckpt


def kd_loss_fn(student, teacher, proj, T, alpha, beta):
    """Returns loss_fn(xb, yb) for common.fit."""

    def loss_fn(xb, yb):
        with torch.no_grad():
            t_logits, t_feat = teacher(xb, with_feat=True)
        s_logits, s_feat = student(xb, with_feat=True)
        loss = (1 - alpha) * F.cross_entropy(s_logits, yb)
        if alpha:
            soft = F.kl_div(F.log_softmax(s_logits.float() / T, 1), F.softmax(t_logits.float() / T, 1),
                            reduction="batchmean")
            loss = loss + alpha * T * T * soft
        if beta:
            t_feat = t_feat.float()
            loss = loss + beta * F.mse_loss(proj(s_feat).float(), t_feat) / (t_feat.var() + 1e-6)
        return loss

    return loss_fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=STUDENTS + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    ap.add_argument("--alpha", type=float, default=0.9, help="weight of the soft-target term")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the feature term (0 = off)")
    ap.add_argument("--tag", default="kd", help="checkpoint suffix, e.g. for ablation runs")
    args = ap.parse_args()

    dev = get_device()
    xtr, ytr, xte, yte = (t.to(dev) for t in load_data("cifar100"))
    teacher, meta = load_ckpt(CKPT_DIR / "teacher_resnet50.pth", dev)
    teacher.requires_grad_(False).to(memory_format=torch.channels_last)
    print(f"teacher test top-1: {meta['test_acc']:.2f}%")
    RES_DIR.mkdir(exist_ok=True)

    for name in STUDENTS if args.student == "all" else [args.student]:
        seed_all(0)
        student = build(name, 100)
        proj = nn.Sequential(nn.Conv2d(student.feat_ch, teacher.feat_ch, 1, bias=False), nn.BatchNorm2d(teacher.feat_ch))
        proj.to(dev).to(memory_format=torch.channels_last)
        history, run = [], f"{name}_{args.tag}"
        fit(student, xtr, ytr, args.epochs, args.lr, loss_fn=kd_loss_fn(student, teacher, proj, args.T, args.alpha, args.beta),
            extra=[proj], eval_data=(xte, yte), eval_every=2, tag=run, history=history, resume=True)
        top1, top5 = evaluate(student, xte, yte)
        print(f"{run}  test top-1 {top1:.2f}%  top-5 {top5:.2f}%  ({count_params(student) / 1e6:.2f}M params)\n")
        pd.DataFrame(history).to_csv(RES_DIR / f"log_{run}.csv", index=False)
        save_ckpt(student, CKPT_DIR / f"{run}.pth", name, "cifar100", 100, test_acc=top1,
                  kd=dict(T=args.T, alpha=args.alpha, beta=args.beta))
