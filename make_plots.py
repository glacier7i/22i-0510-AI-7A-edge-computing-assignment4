"""Figures for the report, built from the CSV files in results/.   ->  results/graphs/*.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RES_DIR
from models import STUDENTS

OUT = RES_DIR / "graphs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
log = lambda name: pd.read_csv(RES_DIR / f"log_{name}.csv")
bench = pd.read_csv(RES_DIR / "benchmark.csv").set_index("model")

# 1. teacher training curve
t = log("teacher").dropna(subset=["test_acc"])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(t.epoch, t.test_acc)
ax.set(xlabel="epoch", ylabel="CIFAR-100 test accuracy (%)", title="ResNet-50 teacher")
fig.tight_layout(); fig.savefig(OUT / "fig_teacher_training.png"); plt.close(fig)

# 2. baseline vs distilled training curves
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, s in zip(axes, STUDENTS):
    for kind, label in (("baseline", "baseline"), ("kd", "distilled")):
        d = log(f"{s}_{kind}").dropna(subset=["test_acc"])
        ax.plot(d.epoch, d.test_acc, label=label)
    ax.set(title=s, xlabel="epoch")
axes[0].set_ylabel("CIFAR-100 test accuracy (%)")
axes[0].legend()
fig.tight_layout(); fig.savefig(OUT / "fig_kd_vs_baseline_curves.png"); plt.close(fig)

# 3. final accuracy: teacher, baseline, distilled
fig, ax = plt.subplots(figsize=(7.5, 4))
w = 0.38
for i, s in enumerate(STUDENTS):
    b, k = bench.loc[f"{s}_baseline", "top1"], bench.loc[f"{s}_kd", "top1"]
    ax.bar(i - w / 2, b, w, color="tab:gray", label="baseline" if i == 0 else None)
    ax.bar(i + w / 2, k, w, color="tab:green", label="distilled" if i == 0 else None)
    ax.text(i - w / 2, b + 0.3, f"{b:.1f}", ha="center"); ax.text(i + w / 2, k + 0.3, f"{k:.1f}", ha="center")
ax.axhline(bench.loc["teacher_resnet50", "top1"], color="tab:red", ls="--", label="ResNet-50 teacher")
ax.set(xticks=range(3), xticklabels=STUDENTS, ylabel="CIFAR-100 test accuracy (%)", ylim=(40, 90))
ax.legend(loc="upper center", ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_accuracy_baseline_vs_kd.png"); plt.close(fig)

# 4. accuracy vs CPU latency for the CIFAR-100 models
c100 = bench[(bench.dataset == "cifar100") & bench.ort_cpu_ms.notna()]
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, r in c100.iterrows():
    ax.scatter(r.ort_cpu_ms, r.top1, s=60, marker="o" if name.endswith("_kd") else "s")
    ax.annotate(name, (r.ort_cpu_ms, r.top1), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set(xlabel="ONNX Runtime CPU latency, batch 1, 1 thread (ms)", ylabel="CIFAR-100 test accuracy (%)",
       title="Accuracy vs. speed (circles: distilled, squares: baseline)")
fig.tight_layout(); fig.savefig(OUT / "fig_accuracy_vs_latency.png"); plt.close(fig)

# 5. NAS search
h = pd.read_csv(RES_DIR / "nas_history.csv")
ok, bad = h[h.status == "ok"], h[h.status == "rejected"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(bad.latency_ms, bad.size_mb, c="tab:red", marker="x", label="rejected")
sc = axes[0].scatter(ok.latency_ms, ok.size_mb, c=ok.val_acc, cmap="viridis", label="trained")
plt.colorbar(sc, ax=axes[0], label="val accuracy (%)")
axes[0].set(xlabel="CPU latency (ms)", ylabel="model size (MB)", title="Candidates vs. deployment limits")
axes[0].legend()
axes[1].scatter(ok.trial, ok.val_acc, s=18)
axes[1].plot(ok.trial, ok.val_acc.cummax(), color="tab:orange", label="best so far")
axes[1].set(xlabel="trial", ylabel="val accuracy (%)", title="Search progress"); axes[1].legend()
axes[2].scatter(ok.latency_ms, ok.val_acc, s=18)
axes[2].set(xlabel="CPU latency (ms)", ylabel="val accuracy (%)", title="Accuracy vs. latency (accepted)")
fig.tight_layout(); fig.savefig(OUT / "fig_nas_search.png"); plt.close(fig)

# 6. NAS model final training
n = log("nas_best").dropna(subset=["test_acc"])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(n.epoch, n.test_acc)
ax.set(xlabel="epoch", ylabel="CIFAR-10 test accuracy (%)", title="Best NAS architecture, trained from scratch")
fig.tight_layout(); fig.savefig(OUT / "fig_nas_final_training.png"); plt.close(fig)
print("figures saved to results/graphs/")
