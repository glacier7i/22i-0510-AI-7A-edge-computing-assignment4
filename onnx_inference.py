"""CPU inference with ONNX Runtime for every exported model: accuracy on the test set, latency, sample predictions.
    python onnx_inference.py                       # all models in onnx/
    python onnx_inference.py --model nas_best --n 5
Results are written to results/onnx_results.csv"""
import argparse

import numpy as np
import pandas as pd

from common import CKPT_DIR, ONNX_DIR, RES_DIR, class_names, load_data
from hardware import ort_latency_ms, ort_session
from models import load_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--model", help="only this model (name without .onnx)")
ap.add_argument("--n", type=int, default=3, help="sample predictions to print per model")
args = ap.parse_args()

data, rows = {}, []
for path in sorted(ONNX_DIR.glob("*.onnx")):
    if path.stem.startswith("_") or (args.model and path.stem != args.model):
        continue
    meta = load_ckpt(CKPT_DIR / f"{path.stem}.pth")[1]
    ds = meta["dataset"]
    if ds not in data:
        data[ds] = load_data(ds)[2:]
    xte, yte = data[ds]

    sess = ort_session(path, threads=0)   # 0 = let ONNX Runtime pick; only the latency run below is pinned to 1 thread
    logits = np.concatenate([sess.run(None, {"input": xte[i:i + 250].numpy()})[0] for i in range(0, len(xte), 250)])
    pred = logits.argmax(1)
    acc = 100 * (pred == yte.numpy()).mean()
    lat = ort_latency_ms(ort_session(path, 1))
    rows.append({"model": path.stem, "dataset": ds, "onnx_top1": round(acc, 2), "ort_cpu_latency_ms": round(lat, 3)})
    print(f"{path.stem:<28} ONNX Runtime CPU: top-1 {acc:.2f}%  latency {lat:.2f} ms (1 thread)")
    names = class_names(ds)
    for j in range(args.n):
        print(f"    image {j}: true {names[yte[j]]:<14} predicted {names[pred[j]]}")

RES_DIR.mkdir(exist_ok=True)
if not args.model:
    pd.DataFrame(rows).to_csv(RES_DIR / "onnx_results.csv", index=False)
