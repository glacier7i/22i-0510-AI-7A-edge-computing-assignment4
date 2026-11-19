"""ONNX export and CPU measurements (latency, file size, memory) used by the NAS, the ONNX scripts and the benchmark."""
import threading
import time
import warnings

import numpy as np
import onnxruntime as ort
import psutil
import torch
import torch.nn as nn

from common import best_of

warnings.filterwarnings("ignore", category=UserWarning)
# tracing treats the python flag with_feat=False in model.forward as a constant, which is what we want for export
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
ort.set_default_logger_severity(3)


def export_onnx(model, path, size=32):
    """Export a model to ONNX with a dynamic batch axis (TorchScript-based exporter)."""
    path.parent.mkdir(exist_ok=True)
    model = model.cpu().eval()
    torch.onnx.export(model, (torch.zeros(1, 3, size, size),), str(path), dynamo=False, opset_version=18,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    return path


def ort_session(path, threads=1):
    """CPU-only ONNX Runtime session. One thread by default, which is the usual edge-device setting."""
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_latency_ms(sess, batch=1, n=100, warmup=20, rounds=3):
    """Median ONNX Runtime CPU latency of one forward pass in ms (best of a few rounds)."""
    x = np.random.randn(batch, 3, 32, 32).astype(np.float32)
    return best_of(lambda: sess.run(None, {"input": x}), rounds=rounds, n=n, warmup=warmup)


def torch_latency_ms(model, batch=1, n=100, warmup=20, rounds=3):
    """Median PyTorch CPU latency in ms. Uses the same thread count as the current torch setting."""
    model = model.cpu().eval()
    x = torch.randn(batch, 3, 32, 32)
    with torch.no_grad():
        return best_of(lambda: model(x), rounds=rounds, n=n, warmup=warmup)


def peak_rss_mb(sess, runs=30):
    """Extra resident memory of this process while running inference, sampled every millisecond.
    Approximate (it is a whole-process number), so it is reported next to the exact activation size below."""
    proc, x = psutil.Process(), np.random.randn(1, 3, 32, 32).astype(np.float32)
    sess.run(None, {"input": x})
    base, peak, stop = proc.memory_info().rss, [0], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.001)

    t = threading.Thread(target=sample)
    t.start()
    for _ in range(runs):
        sess.run(None, {"input": x})
    stop.set()
    t.join()
    return max(peak[0] - base, 0) / 2 ** 20


@torch.no_grad()
def activation_mb(model, size=32):
    """Largest input + output tensor pair (float32, batch 1) over all layers: the working memory a layer-by-layer
    runtime needs for activations. Deterministic, unlike the RSS measurement."""
    model, sizes = model.cpu().eval(), []
    hooks = [m.register_forward_hook(lambda m, i, o: sizes.append(i[0].numel() + o.numel()))
             for m in model.modules() if not list(m.children())]
    model(torch.zeros(1, 3, size, size))
    [h.remove() for h in hooks]
    return max(sizes) * 4 / 2 ** 20


def file_mb(path):
    return path.stat().st_size / 2 ** 20
