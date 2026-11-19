"""Benchmark every checkpoint: accuracy, size, compute and measured CPU speed.   ->  results/benchmark.csv

Accuracy is measured with PyTorch on the test set. Latency is measured on the CPU with one thread and batch size 1
(PyTorch eager and ONNX Runtime); throughput uses ONNX Runtime with batch 64. File sizes are the real files on disk.
"""
import numpy as np
import pandas as pd
import torch

from common import CKPT_DIR, ONNX_DIR, RES_DIR, count_macs, count_params, evaluate, get_device, load_data, median_ms
from hardware import activation_mb, file_mb, ort_latency_ms, ort_session, peak_rss_mb, torch_latency_ms
from models import load_ckpt

torch.set_num_threads(1)
dev, data, rows = get_device(), {}, []
order = lambda p: (p.stem.startswith("teacher") is False, p.stem.endswith("baseline") is False, p.stem)

for ck in sorted(CKPT_DIR.glob("*.pth"), key=order):
    model, meta = load_ckpt(ck, dev)
    ds = meta["dataset"]
    if ds not in data:
        data[ds] = load_data(ds)[2:]
    top1, top5 = evaluate(model, *data[ds], dev)
    model = model.cpu()
    row = {"model": ck.stem, "dataset": ds, "top1": round(top1, 2), "top5": round(top5, 2),
           "params_M": round(count_params(model) / 1e6, 3), "macs_M": round(count_macs(model) / 1e6, 1),
           "checkpoint_MB": round(file_mb(ck), 2), "act_mem_MB": round(activation_mb(model), 3),
           "torch_cpu_ms": round(torch_latency_ms(model, n=50 if "resnet" in meta["arch"] else 100), 3)}
    onnx = ONNX_DIR / f"{ck.stem}.onnx"
    if onnx.exists():
        sess = ort_session(onnx)
        x = np.random.randn(64, 3, 32, 32).astype(np.float32)
        row.update(onnx_MB=round(file_mb(onnx), 2), ort_cpu_ms=round(ort_latency_ms(sess), 3),
                   ort_throughput_img_s=round(64e3 / median_ms(lambda: sess.run(None, {"input": x}), n=20, warmup=5)),
                   peak_rss_MB=round(peak_rss_mb(sess), 2))
    rows.append(row)
    print(row, flush=True)

RES_DIR.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(RES_DIR / "benchmark.csv", index=False)
print("\nsaved results/benchmark.csv")
