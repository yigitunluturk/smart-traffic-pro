# SMART TRAFFIC PRO — Single-Camera Speed Estimation & Turkish ALPR

**SEN499 Graduation Project (2026–2027 Fall)** · Yiğit Ünlütürk · License: [MIT](LICENSE)

SMART TRAFFIC PRO is a single-camera intelligent transportation system. It detects and tracks vehicles,
estimates their speed on a calibrated road plane, reads Turkish licence plates, and records speeding
violations with evidence snapshots. Plates of non-violating vehicles are blurred (KVKK/GDPR).
Inference runs on a standard **CPU**; a GPU is only used (optionally, on Colab) for detector fine-tuning.

> The prototype frozen at Git tag `v0-baseline` is the starting point; only work after that tag is evaluated.
> The requirement-by-requirement status is given in the RSA document.

---

## Architecture

```
Presentation   app.py  (Streamlit — the only official UI)      main.py (CLI)
                    │
Business logic  src/pipeline.py ─┬─ detector.py          YOLO11m + BoT-SORT
                                 ├─ camera_motion.py     camera-drift compensation (ORB + RANSAC)
                                 ├─ speed_estimator.py   road-plane homography + Theil–Sen speed fit
                                 ├─ plate_recognizer.py  FastALPR + Turkish syntax validation + temporal voting
                                 ├─ privacy.py           plate blurring for non-violating vehicles
                                 ├─ violation_manager.py violation records & evidence snapshots
                                 └─ visualizer.py        HUD / overlays
Data access     src/database.py   SQLite (transits, sessions, plate-correction log, retention)
Evaluation      scripts/          eval_speed.py, eval_alpr.py, eval_detector.py, benchmark_fps.py, …
```

## Use cases (Streamlit screens)

| UC | Screen | Requirements |
|----|--------|--------------|
| UC-01 Calibrate camera / speed zone | S1 Calibration | FR-01 |
| UC-02 Monitor live stream / file run | S2 Live Monitoring | FR-02, FR-05, FR-10 |
| UC-03 Audit & correct plate records | S3 Audit & Correct | FR-04, FR-07, FR-09 |
| UC-04 Export compliance data | S4 Records & Export | FR-08, FR-11 |
| UC-05 Review evaluation results | S5 Evaluation Results | FR-06 |
| UC-06 Manage privacy & retention | S6 Privacy & Retention | FR-07, NFR-5 |

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py            # http://localhost:8501
```

Docker (CPU):

```bash
docker compose up --build       # http://localhost:8501
```

CLI (no UI):

```bash
python main.py --input path/to/video.mp4 --output output/result.mp4
```

### Calibrating a new camera (UC-01)

Speed is only accurate for a calibrated camera. Either use the **S1 Calibration** screen, or:

```bash
python road_calibrator.py --input path/to/video.mp4 --width 3.5 --length 24
```

Mark the four corners of a road rectangle whose real size is known (for example two lane lines = lane
width, and two dash starts along the road = length). The result is saved to
`config/calibrations/<video-name>.json` and loaded automatically. Without a calibration file the system
falls back to an approximate mode and prints a warning.

---

## Configuration (`config/default_config.yaml`)

| Key | Meaning |
|-----|---------|
| `model.device` | `cpu` (default) or `cuda` |
| `model.imgsz` | detector input size (640 per NFR-3) |
| `model.frame_stride` | analyse every N-th frame (NFR-3); speed uses timestamps, so accuracy is kept |
| `model.tracker` | `botsort.yaml` |
| `speed.method` | `auto` (per-video calibration if present), `homography`, `zones`, `corridor` |
| `plate.plate_format` | `tr` — Turkish syntax, RSA §5.2 |
| `privacy.blur_non_violators` / `privacy.retention_days` | FR-07 / NFR-5 |

---

## Evaluation (FR-06)

Only numbers produced by these scripts are reported. Each script writes `output/eval/<name>.json`,
which is shown on screen S5. No result is claimed before the scripts have run.

| Purpose | Command |
|---------|---------|
| Speed MAE on BrnoCompSpeed (NFR-1) | `python scripts/eval_speed.py --brno <session_dir>` |
| Speed baseline (no perspective calibration) | `python scripts/eval_speed.py --brno <session_dir> --method corridor` |
| Plate accuracy, proposed (NFR-2) | `python scripts/eval_alpr.py --data data/tr_plates` |
| Plate baselines | `--method single` (FastALPR single frame), `--method easyocr` |
| Prepare UA-DETRAC (YOLO format, fixed split) | `python scripts/prepare_ua_detrac.py --root <UA-DETRAC> --out data/ua_detrac` |
| Fine-tune YOLO11m (Colab GPU) | `python scripts/train_detector.py --data data/ua_detrac/ua_detrac.yaml` |
| Detector comparison (mAP@50) | `python scripts/eval_detector.py --data data/ua_detrac/ua_detrac.yaml` |
| CPU throughput (NFR-3) | `python scripts/benchmark_fps.py --video <1080p.mp4> --stride 2` |
| UA-DETRAC frames → demo video | `python scripts/frames_to_video.py --frames <seq_dir> --out demo.mp4` |

Datasets, licences and KVKK handling are described in [docs/DATASETS.md](docs/DATASETS.md).

Project documents: [Updated RSA](docs/RSA_UPDATED.pdf) · [Architecture and Design Draft](docs/ARCHITECTURE_AND_DESIGN_DRAFT.pdf) · [AI usage declaration](docs/AI_USAGE_DECLARATION.md).

## Tests

```bash
pytest tests/ -v --cov=src
```

GitHub Actions (`.github/workflows/ci.yml`) runs the tests on every push and pull request (CPU only).

## Repository layout

```
app.py                Streamlit dashboard (official UI)
main.py               CLI runner
road_calibrator.py    road-plane calibration tool (OpenCV window)
zone_calibrator.py    entry/exit polygon tool (v0 "zones" method)
config/               default_config.yaml, calibrations/<video>.json
src/                  business logic + data access
scripts/              evaluation, dataset preparation, training
tests/                unit and integration tests
docs/                 project documents (RSA, design draft; docs/archive = superseded drafts)
legacy/               v0 Flask web panel — unsupported, kept for reference only
```
