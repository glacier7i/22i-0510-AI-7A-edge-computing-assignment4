"""Hardware-aware neural architecture search with Optuna (TPE, a Bayesian optimisation method) on CIFAR-10.

For every sampled architecture:
  1. build it, export it to ONNX and measure CPU latency, file size and memory with ONNX Runtime;
  2. reject it right away if it breaks a deployment limit (it is never trained);
  3. otherwise train it for a few epochs and score it by validation accuracy.
TPE receives the limit violations as constraints, so it learns to stay inside the feasible region.

Outputs: results/nas_history.csv, results/nas_history.json, results/nas_best.json
"""
import argparse
import json
import time

import optuna
import pandas as pd
import torch

from common import ONNX_DIR, RES_DIR, count_params, evaluate, fit, get_device, load_data, seed_all, split_val
from hardware import activation_mb, export_onnx, file_mb, ort_latency_ms, ort_session, peak_rss_mb
from nas_space import build_net, suggest_config

# deployment limits (batch 1, one CPU thread, ONNX Runtime)
LIMITS = {"latency_ms": 1.5, "size_mb": 1.0}
TMP_ONNX = ONNX_DIR / "_candidate.onnx"


def measure(model):
    """Hardware numbers of a candidate, measured on this machine's CPU."""
    export_onnx(model, TMP_ONNX)
    sess = ort_session(TMP_ONNX)
    return {"latency_ms": ort_latency_ms(sess), "size_mb": file_mb(TMP_ONNX),
            "peak_rss_mb": peak_rss_mb(sess), "act_mb": activation_mb(model), "params": count_params(model)}


def violated(hw):
    return {k: hw[k] > v for k, v in LIMITS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=8, help="proxy training epochs per candidate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    dev = get_device()
    xtr, ytr, _, _ = load_data("cifar10")
    xtr, ytr, xval, yval = split_val(xtr, ytr, 5000)   # the test set is not touched during the search
    xtr, ytr, xval, yval = xtr.to(dev), ytr.to(dev), xval.to(dev), yval.to(dev)
    RES_DIR.mkdir(exist_ok=True)

    def objective(trial):
        cfg = suggest_config(trial)
        model = build_net(cfg, 10)
        hw = measure(model)
        bad = violated(hw)
        trial.set_user_attr("hw", hw)
        # constraint value > 0 means violated; relative amount so both limits are comparable
        trial.set_user_attr("constraints", [hw[k] / LIMITS[k] - 1 for k in LIMITS])
        trial.set_user_attr("config", cfg)
        if any(bad.values()):
            trial.set_user_attr("status", "rejected")
            print(f"trial {trial.number:>3}  REJECTED  latency {hw['latency_ms']:.2f} ms  size {hw['size_mb']:.2f} MB", flush=True)
            return 0.0
        t0 = time.time()
        fit(model, xtr, ytr, args.epochs, 0.1, bs=256, verbose=False)
        acc = evaluate(model, xval, yval, dev)[0]
        trial.set_user_attr("status", "ok")
        print(f"trial {trial.number:>3}  val_acc {acc:.2f}%  latency {hw['latency_ms']:.2f} ms  "
              f"size {hw['size_mb']:.2f} MB  ({time.time() - t0:.0f}s)", flush=True)
        return acc

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=15,
                                         constraints_func=lambda t: t.user_attrs["constraints"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="nas",
                                storage=f"sqlite:///{RES_DIR / 'nas_study.db'}", load_if_exists=True)
    done = len(study.trials)
    if done:
        print(f"resuming: {done} trials already in the study")
    study.optimize(objective, n_trials=max(args.trials - done, 0))

    rows = [{"trial": t.number, "status": t.user_attrs["status"], "val_acc": t.value if t.user_attrs["status"] == "ok" else None,
             **t.user_attrs["hw"], "config": json.dumps(t.user_attrs["config"])} for t in study.trials]
    hist = pd.DataFrame(rows)
    hist.drop(columns="config").to_csv(RES_DIR / "nas_history.csv", index=False)
    (RES_DIR / "nas_history.json").write_text(json.dumps(rows, indent=1))

    # best = highest accuracy; among candidates within 0.3 points of it, the fastest one
    ok = hist[hist.status == "ok"]
    near = ok[ok.val_acc >= ok.val_acc.max() - 0.3]
    best = near.sort_values("latency_ms").iloc[0]
    out = {"trial": int(best.trial), "val_acc": best.val_acc, "limits": LIMITS,
           "hardware": {k: float(best[k]) for k in ("latency_ms", "size_mb", "peak_rss_mb", "act_mb", "params")},
           "config": json.loads(best.config)}
    (RES_DIR / "nas_best.json").write_text(json.dumps(out, indent=1))
    print(f"\n{len(ok)} accepted, {len(hist) - len(ok)} rejected of {len(hist)} candidates")
    print("best:", json.dumps(out, indent=1))
    TMP_ONNX.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
