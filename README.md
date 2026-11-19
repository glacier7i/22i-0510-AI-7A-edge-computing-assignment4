# Assignment 4: Knowledge Distillation and Hardware-Aware NAS

Hamid Ashraf, Roll No. 22i-0510, Section 7A (Edge Computing & Intelligent Systems)

**Question 1.** A ResNet-50 teacher is trained on CIFAR-100 and distilled into three lightweight students
(MobileNetV2, SqueezeNet, EfficientNet-Lite0). Each student is also trained without distillation as a baseline.
All students are exported to ONNX and run on the CPU with ONNX Runtime.

**Question 2.** An Optuna (TPE) architecture search on CIFAR-10 finds a small CNN that stays inside measured CPU latency
and model-size limits. The best architecture is trained from scratch, exported to ONNX and run with ONNX Runtime.

Everything is plain PyTorch. The report is in `report/`.

## Results

CIFAR-100 test set (10,000 images), 50 epochs per student, same recipe with and without distillation:

| Model | Top-1 | Top-5 | Params | ONNX Runtime CPU latency |
|---|---|---|---|---|
| ResNet-50 teacher | 79.71 % | 95.18 % | 23.71 M | not exported |
| MobileNetV2, baseline | 73.51 % | 93.52 % | 2.35 M | 1.61 ms |
| MobileNetV2, distilled | 76.66 % | 94.86 % | 2.35 M | 1.58 ms |
| SqueezeNet, baseline | 66.73 % | 90.31 % | 0.78 M | 0.39 ms |
| SqueezeNet, distilled | 68.10 % | 90.94 % | 0.78 M | 0.40 ms |
| EfficientNet-Lite0, baseline | 72.75 % | 93.18 % | 3.50 M | 1.81 ms |
| EfficientNet-Lite0, distilled | 76.12 % | 94.73 % | 3.50 M | 1.81 ms |

CIFAR-10 test set, architecture found by the search (limits: 1.5 ms and 1.0 MB):

| Model | Top-1 | Params | ONNX size | ONNX Runtime CPU latency |
|---|---|---|---|---|
| NAS model | 91.04 % | 60,426 | 0.23 MB | 0.53 ms |

Latency is the median of 100 single-image runs on one CPU thread (best of three rounds). The numbers move by roughly
10 to 30 percent from run to run on a laptop CPU, so differences of that size between two models should not be read
as real. A distilled student has exactly the same architecture as its baseline, so any latency difference between the two is
measurement noise. All numbers come from `results/benchmark.csv`.

## Requirements
- Python 3.11 with `torch` 2.10, `torchvision` 0.25, `onnx` 1.23, `onnxruntime` 1.31, `optuna` 4.7, `pandas`, `matplotlib`,
  `numpy`, `psutil`, `pillow`.
- The datasets in `data/` (`cifar-100-python/` and `cifar-10-batches-py/`). The data folder is not part of the repository.
  If a dataset is missing, the first script that needs it downloads it with torchvision (about 170 MB each, needs internet).
- A CUDA GPU for training (the scripts fall back to the CPU, which is far slower). All latency measurements run on the CPU.
- Checkpoints are plain `state_dict` files with the settings needed to rebuild the model, so they load without any extra files.

## How to run
Run these from this folder, in this order:

```bash
python train_teacher.py       # ResNet-50 on CIFAR-100          -> checkpoints/teacher_resnet50.pth
python train_baseline.py      # three students, no distillation -> checkpoints/<student>_baseline.pth
python distill.py             # three students, distilled       -> checkpoints/<student>_kd.pth
python nas_search.py          # architecture search (CIFAR-10)  -> results/nas_history.csv, nas_best.json
python nas_final.py           # best architecture from scratch  -> checkpoints/nas_best.pth
python export_onnx.py         # all students + NAS model        -> onnx/*.onnx
python onnx_inference.py      # ONNX Runtime CPU inference      -> results/onnx_results.csv
python benchmark.py           # accuracy, size, latency, memory -> results/benchmark.csv
python make_plots.py          # figures                         -> results/graphs/
```

`pipeline.ipynb` runs the same steps in order and shows their output. A training step is skipped when its checkpoint
already exists, so "Run All" takes a few minutes; set `RETRAIN = True` in the first cell to train everything again
(about 1 hour for the teacher, 1.5 hours for the six students and 30 to 40 minutes for the search and the final NAS model
on an RTX 4060 laptop GPU).

Useful options:
- `python train_baseline.py --student mobilenetv2 --epochs 50` and the same flags for `distill.py` train a single student.
- `python distill.py --T 4 --alpha 0.9 --beta 1.0` sets the temperature, the weight of the soft-target term and the weight of the
  feature term. `--alpha 0` removes the soft term and `--beta 0` removes the feature term; `--tag name` changes the checkpoint name.
- `python nas_search.py --trials 60 --epochs 10` sets the number of candidates and the proxy training epochs. The study is stored in
  `results/nas_study.db`, so an interrupted search continues where it stopped.
- Training runs save `checkpoints/resume_<run>.pt` after each epoch. If a run is interrupted, start the same command again and
  it continues from the last finished epoch. The file is removed when the run completes.

Inference on any trained model:

```bash
python inference.py --list
python inference.py --model mobilenetv2_kd --n 5          # random test images
python inference.py --model nas_best --image my_photo.png # one image file
python inference.py --model squeezenet_baseline --all     # whole test set
```

## Files

| File | Content |
|---|---|
| `common.py` | data loading, GPU augmentation, training loop, evaluation, timing helpers |
| `models.py` | ResNet-50 teacher, MobileNetV2, SqueezeNet, EfficientNet-Lite0, checkpoint save / load |
| `train_teacher.py`, `train_baseline.py`, `distill.py` | teacher, baseline students, knowledge distillation |
| `nas_space.py` | search space and architecture generator |
| `nas_search.py`, `nas_final.py` | hardware-aware Optuna search, training of the best architecture |
| `hardware.py` | ONNX export and CPU measurements (latency, file size, memory) |
| `inference.py` | one inference script for every model |
| `export_onnx.py`, `onnx_inference.py` | ONNX export, ONNX Runtime CPU inference |
| `benchmark.py`, `make_plots.py` | benchmark table and figures |
| `checkpoints/` | trained models (the teacher is stored with float16 weights to keep the file small) |
| `onnx/` | exported ONNX models |
| `results/` | training logs, NAS history, benchmark CSVs, figures |
| `report/` | technical report |

## Notes
- Models are trained from scratch; no pretrained weights are used.
- The test set is never used to choose a checkpoint or a setting. Every model is the last epoch of its run.
  For CIFAR-100 the training uses all 50,000 training images. The search uses a 45,000 / 5,000 train / validation split of CIFAR-10.
- The strides of MobileNetV2 and EfficientNet-Lite0 were reduced for 32x32 inputs, and SqueezeNet has batch normalization.
  The report lists the exact changes.
- Peak inference memory is reported two ways: the extra resident memory of the process while ONNX Runtime runs
  (rounded, very small for these models) and the largest input-plus-output activation of any layer.
