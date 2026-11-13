# AI Model Evaluation Protocol and Baselines

This document describes **how** the models are evaluated. It deliberately contains **no result numbers**.
Results are reported only after the scripts in `scripts/` have produced them (RSA §1.3, item 3). The
generated files are in `output/eval/*.json` and are shown on dashboard screen S5.

## 1. Vehicle detection (FR-03)

| | Baseline(s) | Proposed |
|---|---|---|
| Model | stock pre-trained `yolo11m.pt` (v0); stock `yolov8n.pt` (lightweight reference) | YOLO11m fine-tuned on UA-DETRAC (`models/yolo11m_uadetrac.pt`) |
| Data | UA-DETRAC test split (see `docs/DATASETS.md`) | same |
| Metric | mAP@50 and mAP@50-95 on vehicle classes (car, bus, truck) | target: +8 points mAP@50 over stock YOLO11m |
| Command | `python scripts/eval_detector.py --data data/ua_detrac/ua_detrac.yaml` | |

Training: `scripts/train_detector.py` (Google Colab GPU, `seed=0`, `deterministic=True`). Inference stays on CPU.

## 2. Plate recognition (FR-04, NFR-2)

| Method | Description | Command |
|---|---|---|
| (a) EasyOCR | EasyOCR on the plate crop | `scripts/eval_alpr.py --method easyocr` |
| (b) FastALPR single frame | raw FastALPR output, no validation | `--method single` |
| Proposed | FastALPR + Turkish syntax validation/correction + temporal voting across frames | `--method proposed` |

Metrics: full-plate accuracy (FPA, all characters correct), character recognition accuracy (CRA,
Levenshtein-based), misread rate (wrong non-empty output), reject rate (no output). Target: FPA ≥ 90 %.
The misread rate of (b) is measured first and then compared with the proposed pipeline.

## 3. Speed estimation (NFR-1)

| Method | Description | Command |
|---|---|---|
| Baseline | naive pixel displacement inside a band, no perspective calibration (`corridor`) | `scripts/eval_speed.py --brno <dir> --method corridor` |
| Proposed | road-plane homography from calibration + camera-drift compensation + Theil–Sen fit over ≥ 0.6 s | `--method homography` |

Data: BrnoCompSpeed. The dataset's own calibration (two vanishing points + scale) is converted
automatically into the road-plane homography. Vehicles are matched to ground truth by passing time
(±1 s). Metrics: MAE, RMSE, median absolute error, mean relative error, recall of matched vehicles. Target MAE ≤ 2.0 km/h.
As the advisor noted, this target is ambitious; measured values are reported as they are.

## 4. Throughput (NFR-3)

`scripts/benchmark_fps.py --video <1080p video> --stride N` measures pipeline FPS on CPU at `imgsz=640`
and records the CPU model and core count.
