"""Export every trained student (and the NAS model) to ONNX.   ->  onnx/<name>.onnx
The teacher is only used for training and is not exported."""
import numpy as np
import torch

from common import CKPT_DIR, ONNX_DIR
from hardware import export_onnx, file_mb, ort_session
from models import load_ckpt

for ck in sorted(CKPT_DIR.glob("*.pth")):
    if ck.stem.startswith("teacher"):
        continue
    model, meta = load_ckpt(ck)
    path = export_onnx(model, ONNX_DIR / f"{ck.stem}.onnx")
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        diff = np.abs(model(x).numpy() - ort_session(path).run(None, {"input": x.numpy()})[0]).max()
    print(f"{ck.stem:<28} -> {path.name:<28} {file_mb(path):6.2f} MB   max |torch - onnx| = {diff:.1e}")
