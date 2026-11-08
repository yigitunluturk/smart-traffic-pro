# 🚗 SMART TRAFFIC PRO

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLO11m-orange.svg)](https://docs.ultralytics.com/)
[![OCR Engine](https://img.shields.io/badge/fast--alpr-ONNX%20Runtime-green.svg)](https://github.com/ankandrew/fast-alpr)
[![Testing](https://img.shields.io/badge/Tests-14%20Passed%20(Pytest)-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](Dockerfile)

**SEN499 Graduation Project (2026–2027 Fall Semester)**  
An end-to-end intelligent transportation and surveillance system that concurrently performs multi-class vehicle detection, persistent trajectory tracking, high-precision timeline-based speed estimation, automated license plate recognition (ALPR), and speeding violation enforcement with automated photographic evidence capture.

---

## 🏛️ System Architecture & Core Use Cases

```text
┌────────────────────────────────────────────────────────┐
│              Presentation Layer (UI / CLI)             │
│        Streamlit Dashboard (app.py) | CLI (main.py)    │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                 Pipeline Orchestration                 │
│                 TrafficPipeline (src/)                 │
└───────┬──────────────┬──────────────┬───────────┬──────┘
        │              │              │           │
┌───────▼──────┐┌──────▼──────┐┌──────▼─────┐┌────▼──────┐
│   Detector   ││    Speed    ││   Plate    ││Violation  │
│   (YOLO11)   ││  Estimator  ││ Recognizer ││  Manager  │
└──────────────┘└─────────────┘└────────────┘└────┬──────┘
                                                  │
┌─────────────────────────────────────────────────▼──────┐
│            Data Persistence Layer (SQLite & CSV)       │
│          TrafficDatabase (output/traffic.db)           │
└────────────────────────────────────────────────────────┘

---

## ▶️ Running

The **Streamlit dashboard (`app.py`) is the only official user interface.**

```bash
pip install -r requirements.txt
streamlit run app.py
```

With Docker:

```bash
docker compose up --build
# then open http://localhost:8501
```

## 📁 Project Status

- `v0-baseline` (Git tag) marks the prototype that existed before this semester. Only work after this tag is evaluated.
- Evaluation scripts (`scripts/eval_speed.py`, `scripts/eval_alpr.py`) will be added this semester. **No accuracy results are reported until these scripts have produced them.**
- `legacy/` contains the earlier Flask web panel (`server.py`, `web/`). It is **unsupported** and not part of the evaluated system.
