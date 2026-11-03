# 🚗 AI-Powered Traffic Surveillance: Real-Time Speed Estimation & ALPR

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLO11m-orange.svg)](https://docs.ultralytics.com/)
[![OCR Engine](https://img.shields.io/badge/fast--alpr-ONNX%20Runtime-green.svg)](https://github.com/ankandrew/fast-alpr)
[![Testing](https://img.shields.io/badge/Tests-14%20Passed%20(Pytest)-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**SEN499 Graduation Project (2026–2027 Fall Semester)**  
An end-to-end intelligent transportation and surveillance system that concurrently performs multi-class vehicle detection, persistent trajectory tracking, high-precision timeline-based speed estimation, automated license plate recognition (ALPR), and speeding violation enforcement with automated photographic evidence capture.

---

## 🏛️ System Architecture & 6 Core Use Cases

```
┌────────────────────────────────────────────────────────┐
│             Presentation Layer (UI / CLI)              │
│       Streamlit Dashboard (app.py) | CLI (main.py)     │
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
│           Data Persistence Layer (SQLite & CSV)        │
│          TrafficDatabase (output/traffic.db)           │
└────────────────────────────────────────────────────────┘
```

1. **UC-1: Multi-Class Vehicle Detection & Tracking:** Detecting cars, buses, trucks, and motorcycles in each video frame using **YOLO11m** and maintaining unique temporal IDs using **BoT-SORT**.
2. **UC-2: Frame-Accurate Speed Estimation:** Computing vehicle velocity based on the camera's fixed framerate timeline between calibrated entry and exit polygons:
   $$\Delta t = \frac{\text{Exit Frame} - \text{Entry Frame}}{\text{Video FPS}}, \quad v = \left(\frac{d_{\text{real}}}{\Delta t}\right) \times 3.6 \times \left(\frac{\Delta y_{\text{measured}}}{\Delta y_{\text{calib}}}\right)$$
   *(Decoupled from inference lag — 100% mathematically exact even on low-spec hardware).*
3. **UC-3: Deep Learning License Plate Recognition:** Localizing license plates using **YOLOv9-t** and performing OCR character recognition using Compact Convolutional Transformers (**CCT-S-v2**).
4. **UC-4: Candidate Accumulation & Confidence Locking:** Stabilizing OCR by collecting candidate readings within the detection band and locking the highest-confidence valid plate to eliminate visual jitter.
5. **UC-5: Speeding Violation Enforcement:** Automatically triggering violation alerts, marking vehicle bounding boxes in red, and capturing high-resolution evidence snapshots (`output/ihlaller/`).
6. **UC-6: Persistent Relational Storage & Analytics:** Logging all sessions, vehicles, and violations into an ACID-compliant **SQLite database** (`output/traffic_surveillance.db`) with CSV export and real-time dashboard visualization.

---

## 📁 Repository Directory Structure

```text
bitirme/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Automated GitHub Actions CI test & lint pipeline
├── config/
│   └── default_config.yaml           # Centralized configuration (models, zones, limits, thresholds)
├── docs/                             # Academic & Graduation Deliverables
│   ├── PROJECT_PROPOSAL_FORM.md      # Appendix-2: Official Proposal Form (Due 09.10.2026)
│   ├── SYSTEM_REQUIREMENTS_AND_ARCHITECTURE.md # ISO/IEC/IEEE 29148 Specification & UML
│   ├── AI_MODEL_EVALUATION_AND_BASELINE.md     # AI Benchmark (YOLO11 vs Baselines, fast-alpr vs OCR)
│   ├── AI_USAGE_DECLARATION.md       # Section 8: Academic Integrity & AI Ethics Statement
│   └── PRESENTATION_AND_DEFENSE_GUIDE.md       # 10-Minute Presentation Script & Jury Q&A
├── output/                           # Generated results (gitignored)
│   ├── ihlaller/                     # Speeding violation evidence snapshots (.jpg)
│   ├── arac_kayitlari.csv            # Tabular vehicle log report
│   └── traffic_surveillance.db       # Persistent SQLite database
├── src/                              # Core Business Logic Layer
│   ├── __init__.py
│   ├── database.py                   # SQLite relational persistence layer
│   ├── detector.py                   # YOLO11m vehicle detection & BoT-SORT tracker
│   ├── speed_estimator.py            # Multi-lane speed calculation & auto-resolution scaling
│   ├── plate_recognizer.py           # fast-alpr OCR engine & confidence locking
│   ├── violation_manager.py          # Violation detection, snapshot exporter & data synchronization
│   ├── visualizer.py                 # Dynamic resolution scaling, HUD overlay & frame annotations
│   └── pipeline.py                   # End-to-end synchronized pipeline coordinator
├── tests/                            # Automated Pytest Suite (14 Tests)
│   ├── __init__.py
│   ├── test_database.py              # SQLite storage unit tests
│   ├── test_speed_estimator.py       # Speed estimation mathematics & scaling tests
│   ├── test_plate_recognizer.py      # ALPR normalization & candidate locking tests
│   ├── test_violation_manager.py     # Violation manager & snapshot tests
│   └── test_integration.py           # End-to-end pipeline integration scenarios
├── app.py                            # Streamlit Web Surveillance Dashboard
├── main.py                           # CLI video processing & live OpenCV runner
├── zone_calibrator.py                # Interactive mouse-driven lane polygon calibrator
├── BASLAT_WEB_PANEL.bat              # 1-Click Windows launcher for Web Dashboard
├── BASLAT_CLI_TEST.bat               # 1-Click Windows launcher for CLI video runner
├── SERIT_KALIBRE_ET.bat              # 1-Click Windows launcher for Zone Calibrator
├── Dockerfile                        # Production container image definition
├── docker-compose.yml                # Single-command container deployment
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Quick Start Guide

### Option 1: Native Python Environment

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Web Surveillance Dashboard (Streamlit):**
   ```bash
   streamlit run app.py
   # Or simply double-click BASLAT_WEB_PANEL.bat on Windows
   ```
   Open your browser at `http://localhost:8501`.

3. **Run via Command Line Interface (CLI):**
   ```bash
   python main.py --input "C:\Users\msi-nb\Desktop\part1.mp4" --output output/result.mp4 --show
   ```

4. **Calibrate New Camera Zones (Interactive Mouse GUI):**
   ```bash
   python zone_calibrator.py --input video.mp4
   # Left-click to add polygon points, 's' to save zone, 'q' to print JSON coordinates
   ```

---

### Option 2: Docker Container Deployment (Cross-Platform)

```bash
# Build and run with single command
docker compose up --build
```
Navigate to `http://localhost:8501` in your browser. All generated snapshots and database records will be saved directly into your local `./output` folder via Docker volume mounting.

---

## 🧪 Automated Testing & CI/CD Quality Gate

Run the automated test suite locally:
```bash
pytest tests/ -v --cov=src -p no:cacheprovider
```

**Test Results:**
- **14 Passed Unit & Integration Tests** (100% pass rate).
- Automated CI pipeline runs via **GitHub Actions** on every `push` and `pull_request` targeting `main`.

---

## 📊 Benchmark Summary (Section 3.3 Compliance)

| Evaluation Metric | Baseline (YOLOv8n / EasyOCR) | Our Solution (YOLO11m / fast-alpr) | Improvement |
| :--- | :---: | :---: | :---: |
| **Vehicle Detection mAP@50** | 68.2% | **78.4%** | **+10.2%** |
| **License Plate Full Read Acc.** | 73.8% | **91.4%** | **+17.6%** |
| **OCR Latency per Crop** | 115 ms | **14 ms** | **8x Faster** |
| **Speed Estimation Error (MAE)** | 3.4 km/h | **1.82 km/h** | **High Precision** |

*(For full benchmark tables and methodology, see [docs/AI_MODEL_EVALUATION_AND_BASELINE.md](docs/AI_MODEL_EVALUATION_AND_BASELINE.md)).*
