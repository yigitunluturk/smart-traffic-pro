# Appendix-2: Project Proposal Form (Ek-2: Proje Önerisi Formu)
**SEN499 Graduation Project · Fall 2026–2027**  
*Submission Deadline: 20.10.2026 | Maximum Length: 2 Pages (Form) + Technical Addendum | Language: English*  
*Project Name: SMART TRAFFIC PRO*

---

| Field | Student's Answer |
| :--- | :--- |
| **Student Name / ID** | *[Insert Your Name]* / *[Insert Your Student ID]* |
| **Project Title** | **SMART TRAFFIC PRO: AI-Powered Real-Time Vehicle Speed Estimation, License Plate Recognition, and Automated Violation Management** |
| **Category** | ☑ **Artificial Intelligence** &nbsp;&nbsp;&nbsp;&nbsp; ☐ Software Development |
| **Problem Statement**<br>*(What problem does it solve? Why does it matter?)* | Modern traffic law enforcement relies heavily on separate, capital-intensive hardware installations: roadside Doppler/RADAR gantries for velocity enforcement and dedicated optical gate cameras for Automated Number Plate Recognition (ANPR). These disparate systems suffer from exorbitant acquisition and maintenance costs, cannot reliably cross-correlate vehicle speeds with license plates under high-density multi-lane conditions, and require intrusive roadwork (e.g., embedded induction loops). SMART TRAFFIC PRO solves this challenge by delivering a single-camera, edge-capable, and non-intrusive intelligent transportation software suite. It seamlessly integrates vehicle detection, persistent multi-object tracking, metric perspective-calibrated velocity estimation, multi-frame temporal license plate recognition, automated photographic violation logging, and a web-based surveillance operations center (SOC). |
| **Target Users** | Municipal traffic monitoring centers, highway law enforcement authorities (e.g., Directorate General of Security / Traffic Police), smart city traffic engineers, toll road concession operators, and automated parking/checkpoint supervisors. |
| **Core Features**<br>*(User Goals with MoSCoW Priority)* | **1. UG-1 [Must Have - M] Calibrating Surveillance Camera & Speed Parameters:** The traffic operator sets up a new or existing camera perspective by interactively drawing entry and exit polygon lines, specifying the ground-truth physical distance between calibrated road markings (in meters), and setting lane-specific speed enforcement thresholds (km/h) for frame-accurate velocity estimation.<br>**2. UG-2 [Must Have - M] Uploading Surveillance Footage & Monitoring Job Execution:** The user uploads recorded surveillance video files (or attaches an RTSP/camera stream) and tracks job progress, live telemetry metrics (FPS throughput, vehicle count, speeding count), and real-time annotated playback.<br>**3. UG-3 [Must Have - M] Reviewing Infractions with Evidence Frames & Plate Crops:** The enforcement officer reviews captured speeding violations in an interactive evidence gallery, inspecting the full-scene evidence frame with timestamp/watermark, cropped vehicle bounding box, zoomed license plate crop, and computed velocity.<br>**4. UG-4 [Should Have - S] Manually Correcting Misread License Plates:** The operator audits and manually corrects low-confidence or wrongly recognized license plate strings directly through the web UI / REST API, persisting the correction with audit traceability.<br>**5. UG-5 [Should Have - S] Querying, Filtering & Auditing Vehicle Transit History:** The operator searches and filters historical transit logs across multiple parameters (plate substring, date/time interval, lane name, violation status) to generate compliance audits and traffic flow summaries.<br>**6. UG-6 [Must Have - M] Exporting Certified Violation Reports & Raw Archives:** The operator exports filtered violation logs and transit records in standard CSV/Excel formats, and downloads the raw SQLite database alongside full-resolution JPEG evidence archives for legal processing.<br>**7. UG-7 [Could Have - C] Real-Time Multi-Modal SOC Violation Alerts:** The live operations room receives instant audio-visual HUD alerts and toast notifications when a monitored vehicle crosses the speed threshold. |
| **Out of Scope**<br>*(What will you NOT build?)* | 1. Vehicle interior cabin passenger surveillance and driver facial identification.<br>2. Direct governmental fine billing, banking, or credit card payment processing integration.<br>3. Road-embedded physical sensor/inductive loop hardware manufacturing. |
| **Additional Scope Items**<br>*(Pick at least two, see Section 3.2)* | ☑ **Real-time features** (Live annotated video stream via MJPEG, instant velocity calculation, HUD telemetry overlays)<br>☑ **Background jobs / Concurrency** (Multi-threaded stream ingestion, asynchronous evidence snapshot generation, and background SQLite database logging)<br>☑ **Containerization** (Full cross-platform Docker & Docker Compose container deployment with CPU/CUDA runtime profiles) |
| **Planned Tech Stack & Concrete Layers**<br>*(Language, framework, database, tools)* | • **Layer 1 - Presentation Layer (UI/Client):** Single-page web dashboard using HTML5, Vanilla CSS3 (modern glassmorphism SOC dark theme), and JavaScript (ES6+ Fetch API, Canvas overlays), plus an interactive Streamlit analytical portal (`app.py`).<br>• **Layer 2 - API & Orchestration Layer (Service/API):** Python 3.11+ Flask RESTful API (`server.py`) serving endpoints for video upload (`/api/upload`), playback control (`/api/control`), live MJPEG feed (`/video_feed`), telemetry (`/api/stats`), record querying (`/api/records`), manual plate correction (`/api/records/<id>/plate`), and file exports (`/api/export/*`).<br>• **Layer 3 - Core AI & Processing Pipeline Layer (Business Logic):** Modular pipeline (`TrafficPipeline`) coordinating YOLO11m vehicle detection with BoT-SORT tracking (`src/detector.py`), perspective-aware polygon speed estimation (`src/speed_estimator.py`), FastALPR ONNX OCR with temporal consensus locking (`src/plate_recognizer.py`), and snapshot/violation logging (`src/violation_manager.py`).<br>• **Layer 4 - Data Persistence Layer (Storage):** SQLite3 ACID relational database (`output/traffic_surveillance.db`) with foreign-key indexed sessions and vehicle records, alongside structured CSV/Excel reports and evidence filesystem storage (`output/ihlaller/`). |
| **Novel Engineering on Top of Existing Components & Code Declaration** | • **Foundational Components Leveraged:** Ultralytics YOLO11 & BoT-SORT (bounding box detection & multi-object tracking) and FastALPR (YOLOv9-t plate detector + CCT-S-v2 character recognition).<br>• **What We Build on Top of Existing Components:**<br>&nbsp;&nbsp;*1. Metric Polygon Zone Calibration & Resolution Invariance:* Ultralytics built-in speed estimation is limited to rudimentary 2-point pixel lines. We engineered a perspective-aware polygon transit engine that scales coordinates across resolutions (720p to 4K), calculates precise physical transit times ($\Delta t$), and maps pixel transit to calibrated ground-truth road markings ($d_{\text{real}}$ in meters) independent of inference framerate fluctuations.<br>&nbsp;&nbsp;*2. Temporal Multi-Frame Plate Accumulation & Consensus Locking:* FastALPR operates per-frame on individual crops, causing jitter/misreads from motion blur or angles. We engineered a temporal candidate buffer (`PlateTrackState`) that aggregates OCR candidates across sequential frames, validates syntax against national/regional regex standards (e.g. TR 81-province standard `\d{2}[A-Z]{1,3}\d{2,4}`), filters low-confidence outliers, and locks the highest-confidence valid string once consensus ($N \ge 3$) is achieved.<br>&nbsp;&nbsp;*3. Full Multi-Layer Surveillance Suite & SOC Web Interface:* Built a comprehensive 4-tier Intelligent Transportation System (ITS) integrating REST APIs, real-time web video streaming, telemetry, evidence snapshot generation, and manual correction.<br>&nbsp;&nbsp;*4. Rigorous Evaluation Setup:* A standardized evaluation benchmark measuring velocity error against radar ground-truth datasets and OCR accuracy across standardized plate datasets.<br>• **Code Declaration:** A functional working prototype has already been developed by the student in this repository (`src/`, `web/`, `app.py`, `server.py`, `zone_calibrator.py`, and `tests/`). This proposal outlines the formalization, model fine-tuning, benchmark evaluation on public ground-truth datasets, and CPU execution hardening of this system. |
| **Datasets, Ground-Truth Speed & Error Metrics**<br>*(AI projects only: dataset, license, baseline)* | • **Vehicle Detection Fine-Tuning:** Trained/fine-tuned on the **UA-DETRAC** (University at Albany DETection and tRACking) dataset (>140k frames across diverse weather, lighting, and congestion conditions, License: *CC BY-NC-SA 4.0*) and COCO Traffic Subset (*CC BY 4.0*).<br>• **Speed Ground-Truth Dataset & Metric:** Evaluated on the **BrnoCompSpeed Dataset** (Brno University of Technology, public academic license), featuring 18 full-HD videos with precise ground-truth velocity labels from calibrated optical/radar traps. The evaluation metric is **Mean Absolute Error (MAE in km/h)**: $\text{MAE} = \frac{1}{N} \sum |v_{\text{pred}} - v_{\text{radar}}|$, targeting $\text{MAE} \le 2.5\text{ km/h}$, alongside RMSE and Mean Relative Error (MRE %).<br>• **Plate Recognition Test Set:** Evaluated on the **UFPR-ALPR Dataset** (Federal University of Paraná, 4,500 annotated vehicle images/tracks across 30-frame video sequences, public academic license) and a regional Turkish license plate benchmark set, evaluated via Character Recognition Accuracy (CRA %) and Full-Plate Matching Accuracy (FPA %). |
| **Hardware Independence & CPU-First Execution**<br>*(Evaluation Computer & GitHub Actions)* | • The system is architected to run out-of-the-box on **standard CPU hardware** without requiring a dedicated GPU.<br>• Ultralytics YOLO executes via PyTorch CPU backend with multi-core threading; FastALPR runs ONNX Runtime with `CPUExecutionProvider`. Video writing defaults to CPU codecs (`mp4v` / `libx264`).<br>• Hardware acceleration (NVIDIA CUDA `--device 0` and NVENC encoder `h264_nvenc`) is implemented strictly as an **optional performance speed-up** with automated fallback to CPU.<br>• Automated testing and CI/CD on **GitHub Actions** (`ubuntu-latest` CPU runner) execute successfully without GPU dependencies (demonstrated by automated Pytest test suites with >60% code coverage). |
| **Privacy & Data Protection Protocol**<br>*(KVKK / GDPR Compliance)* | • **Demo Data:** Live demonstrations, public evaluations, and defenses will strictly utilize videos from public research datasets (BrnoCompSpeed, UA-DETRAC, UFPR-ALPR) licensed for public academic evaluation.<br>• **Anonymization of Field Recordings:** For any locally recorded footage, an automated privacy preservation module applies Gaussian/pixelated blurring to license plates and driver/passenger cabin windows for all non-violating vehicles.<br>• **Data Retention Limit:** Unprocessed raw video recordings will be retained for a maximum of **7 days** on local encrypted storage solely for verification and automatically deleted thereafter via an automated purge script. Only cryptographic evidence snapshots of confirmed infractions are retained for academic evaluation, after which all data is permanently destroyed in full compliance with KVKK (Law No. 6698) and GDPR standards. |
| **Main Risks and Mitigation** | • **Risk 1:** Camera perspective foreshortening distorting speed measurements.<br>*Mitigation:* Interactive polygon calibrator (`zone_calibrator.py`) and metric ground-truth scaling ($d_{\text{real}}$ in meters) with resolution-adaptive coordinate transformation.<br>• **Risk 2:** Motion blur, camera jitter, and temporary plate occlusion at high speeds.<br>*Mitigation:* Multi-frame candidate accumulation buffer with syntax regex filtering and confidence locking over $N \ge 3$ consistent reads.<br>• **Risk 3:** GPU absence in evaluation environment or CI/CD pipelines.<br>*Mitigation:* CPU-first architectural design; lightweight ONNX inference; NVENC/CUDA treated as optional speed-up with zero-crash CPU fallback. |
| **GitHub Username** | *[Insert Your GitHub Username]* (Repository link will be shared with the advisor) |

---

### For the Advisor

| Decision | ☐ Approved &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ☐ Revision Requested &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ☐ Rejected |
| :--- | :--- |
| **Notes** | |
| **Date / Signature** | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; / &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |

---

# Technical Addendum: Detailed Revision Specifications

This addendum provides comprehensive technical specifications addressing the revision feedback points requested for the proposal resubmission (Deadline: 20.10.2026).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SMART TRAFFIC PRO                               │
│              4-Layer System Architecture Specification                 │
└────────────────────────────────────────────────────────────────────────┘

  [LAYER 1: PRESENTATION & CLIENT LAYER]
  ┌──────────────────────────────────────┐  ┌───────────────────────────┐
  │  ArgusVision Web SOC (HTML5/CSS3/JS) │  │ Streamlit Audit Portal    │
  │  - Live MJPEG Stream Player          │  │ - In-depth Table Querying │
  │  - Real-Time KPI Telemetry Cards     │  │ - Evidence Photo Gallery  │
  │  - Interactive Control (Play/Pause)  │  │ - Zone Polygon Visualizer │
  └───────────────────┬──────────────────┘  └─────────────┬─────────────┘
                      │                                   │
  [LAYER 2: APPLICATION & REST API LAYER]                 │
  ┌───────────────────▼───────────────────────────────────▼─────────────┐
  │  Flask REST API Web Service (server.py)                              │
  │  • GET  /video_feed             (Multipart MJPEG live streaming)     │
  │  • POST /api/control            (Stream start / pause / stop)        │
  │  • GET  /api/stats              (Real-time telemetry & counters)     │
  │  • GET  /api/records            (Search & filter transit logs)       │
  │  • PATCH /api/records/<id>/plate (Manual license plate correction)   │
  │  • POST /api/upload             (Surveillance video file upload)     │
  │  • GET  /api/export/csv         (CSV transit & violation export)     │
  │  • GET  /api/export/db          (Raw SQLite database download)       │
  └──────────────────────────────────────┬───────────────────────────────┘
                                         │
  [LAYER 3: CORE AI PIPELINE LAYER]      │
  ┌──────────────────────────────────────▼───────────────────────────────┐
  │  TrafficPipeline (src/pipeline.py)                                   │
  │  ┌────────────────────┐  ┌───────────────────┐  ┌──────────────────┐ │
  │  │  VehicleDetector   │  │  SpeedEstimator   │  │ PlateRecognizer  │ │
  │  │  (YOLO11m+BoTSORT) │  │ (Metric Polygons) │  │(FastALPR+Buffer) │ │
  │  └─────────┬──────────┘  └─────────┬─────────┘  └────────┬─────────┘ │
  │            │                       │                     │           │
  │            └───────────────┬───────┴─────────────────────┘           │
  │                            ▼                                         │
  │                ViolationManager (src/violation_manager.py)           │
  │                - Threshold comparison (speed > speed_limit)          │
  │                - High-res evidence frame & crop capture              │
  └──────────────────────────────────────┬───────────────────────────────┘
                                         │
  [LAYER 4: PERSISTENCE & STORAGE LAYER] │
  ┌──────────────────────────────────────▼───────────────────────────────┐
  │  SQLite3 ACID Database (`output/traffic_surveillance.db`)            │
  │  - `sessions` table (video source, speed limit, start time, totals)  │
  │  - `vehicle_records` table (track_id, plate, speed, snapshot path)   │
  │  Filesystem Evidence: `output/ihlaller/*.jpg` & CSV export           │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 1. User Goals & MoSCoW Prioritization Matrix

| User Goal ID | Role / Actor | User Goal Statement | MoSCoW Priority | Validation Criteria |
| :--- | :--- | :--- | :---: | :--- |
| **UG-1** | Traffic Engineer / Operator | **Calibrating Surveillance Camera:** Define virtual entry/exit polygon lines, input real-world physical distance between road reference markers ($d_{\text{real}}$ in meters), and configure lane speed limits so that vehicle velocities are accurately estimated regardless of camera height or resolution. | **Must Have (M)** | Calibration coordinates persist to `config/default_config.yaml` and auto-scale dynamically across 720p, 1080p, and 4K video feeds. |
| **UG-2** | System Operator | **Uploading Footage & Tracking Job Status:** Upload surveillance video files or attach live RTSP camera feeds, and follow end-to-end processing progress, live frame count, throughput FPS, and telemetry counters in real-time. | **Must Have (M)** | Status endpoint `/api/stats` and progress bar update continuously; stream renders at $\ge 20$ FPS on CPU. |
| **UG-3** | Law Enforcement Officer | **Reviewing Infractions with Evidence:** Inspect captured speeding violations in an interactive evidence gallery, reviewing the full context frame with timestamp watermark, zoomed vehicle crop, recognized plate, and measured velocity. | **Must Have (M)** | Snapshot JPEG saved automatically with vehicle bounding box, plate string, and velocity watermark upon speed limit breach. |
| **UG-4** | Audit Officer / Operator | **Manually Correcting Wrongly Read Plates:** Correct misread, partially occluded, or low-confidence license plates directly through the web UI / REST API to maintain high data fidelity. | **Should Have (S)** | Calling `PATCH /api/records/<id>/plate` updates the database record instantly without requiring pipeline restarts. |
| **UG-5** | Traffic Analyst | **Searching & Auditing Transit Records:** Query historical transit records by plate characters, date/time range, lane identifier, or violation flag to audit traffic density and citation patterns. | **Should Have (S)** | Database queries return filtered results in $<100\text{ ms}$ over 50,000 records using SQLite indexed columns. |
| **UG-6** | Records Administrator | **Exporting Evidence & Reports:** Export structured violation summaries and transit records into standard CSV/Excel sheets and download the raw SQLite database or evidence image bundles. | **Must Have (M)** | Single-click export via `/api/export/csv` and `/api/export/db` generates standard RFC-4180 CSV and intact SQLite database. |
| **UG-7** | SOC Monitoring Operator | **Receiving Real-Time Violation Alerts:** Receive immediate visual badges and audio-visual HUD highlights on the live operations screen whenever a vehicle exceeds the configured speed threshold. | **Could Have (C)** | Video overlay immediately highlights speeding vehicles in red with speed flash tags; live counters increment without latency. |

---

## 2. Novel Engineering on Top of Existing Components

```
┌────────────────────────────────────────────────────────────────────────────────┐
│             WHAT WE BUILD ON TOP OF FOUNDATIONAL MODULES                       │
├────────────────────────────┬───────────────────────────────────────────────────┤
│ Foundation Module          │ Custom Engineered Innovations (SMART TRAFFIC PRO) │
├────────────────────────────┼───────────────────────────────────────────────────┤
│ **Ultralytics YOLO11 &**   │ • **Perspective & Metric Polygon Zone Calibration**:             │
│ **SpeedEstimator**         │   - Replaced basic 2-point pixel lines with multi-point polygons. │
│                            │   - Video framerate timeline interval speed calculation:          │
│                            │     $\Delta t = (\text{Frame}_{\text{exit}} - \text{Frame}_{\text{entry}}) / \text{FPS}$  │
│                            │     $v = (d_{\text{real}} / \Delta t) \times 3.6 \times (\Delta y_{\text{measured}} / \Delta y_{\text{calib}})$ │
│                            │   - Full resolution invariance (auto-scales 720p/1080p/4K).       │
│                            │   - Directional transit verification (filtering opposite traffic).│
├────────────────────────────┼───────────────────────────────────────────────────┤
│ **FastALPR**               │ • **Temporal Multi-Frame Candidate Accumulation Buffer**:         │
│ (YOLOv9-t + CCT-S-v2)      │   - Per-vehicle `PlateTrackState` tracking across sequential frames│
│                            │   - Syntax regex validation (e.g. Turkish 81 provinces format:    │
│                            │     `^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}\d{2,4}$`).             │
│                            │   - Multi-read consensus locking: locks highest confidence string │
│                            │     once $N \ge 3$ consistent reads are achieved, removing jitter.│
├────────────────────────────┼───────────────────────────────────────────────────┤
│ **Evaluation & Benchmark** │ • Ground-truth benchmark harness against **BrnoCompSpeed**       │
│ **Harness**                │   and **UFPR-ALPR** datasets.                                     │
│                            │ • Standardized error metrics: MAE (km/h), RMSE, and CRA/FPA.      │
├────────────────────────────┼───────────────────────────────────────────────────┤
│ **Surveillance System**    │ • Complete 4-layer architecture with Flask REST API & Web SOC.    │
│ **& Operations Center**    │ • Manual plate audit & correction endpoint (`PATCH /api/records`).│
│                            │ • ACID SQLite persistence, CSV reporting, and snapshot evidence.  │
└────────────────────────────┴───────────────────────────────────────────────────┘
```

### Declaration of Existing Codebase
As of this revision submission (20.10.2026), a functional baseline prototype has already been engineered by the student in the project repository:
- `src/detector.py`: YOLO11 multi-class vehicle detection with BoT-SORT tracker.
- `src/speed_estimator.py`: Perspective-aware polygon transit velocity estimation.
- `src/plate_recognizer.py`: FastALPR wrapper with temporal candidate consensus buffer.
- `src/violation_manager.py`: Speed threshold logic, snapshot generator, and CSV exporter.
- `src/database.py`: SQLite relational database layer with manual plate correction support.
- `zone_calibrator.py`: Interactive OpenCV mouse-driven coordinate calibration tool.
- `server.py`: Flask REST API service and MJPEG video streaming controller.
- `web/`: Modern HTML5/CSS/Vanilla JS surveillance operations dashboard.
- `app.py`: Streamlit analytical and investigation center.
- `tests/`: 14 automated unit and integration tests passing with Pytest.

---

## 3. Dataset Specifications, Ground Truth & Evaluation Metrics

### 3.1 Vehicle Detection Fine-Tuning Dataset
- **Dataset:** **UA-DETRAC (University at Albany DETection and tRACking Dataset)**
- **Scope:** 100 video sequences (over 140,000 frames) recorded in high traffic density with varying camera angles, weather conditions (sunny, rainy, cloudy, night), and multi-class annotations (car, bus, truck, van).
- **License:** **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** for academic research.
- **Role:** Fine-tuning YOLO11m weights to maximize vehicle bounding-box recall under road glare, camera shake, and distant perspective foreshortening.

### 3.2 Speed Ground-Truth Dataset & Error Metric
- **Dataset:** **BrnoCompSpeed Dataset** (Brno University of Technology - Faculty of Information Technology)
- **Scope:** 18 full-HD 50/60 FPS traffic surveillance videos recorded from highway and arterial bridge perspectives. Contains over 20,000 vehicle transits with verified ground-truth speed measurements obtained via calibrated optical and Doppler radar traps.
- **License:** Public Academic & Research License.
- **Evaluation Metric:** **Mean Absolute Error (MAE in km/h)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| v_{\text{calculated}, i} - v_{\text{ground\_truth}, i} \right|$$
  Accompanied by **Mean Relative Error (MRE %)** and **Root Mean Square Error (RMSE in km/h)**:
  $$\text{MRE} = \frac{1}{N} \sum_{i=1}^{N} \frac{\left| v_{\text{calculated}, i} - v_{\text{ground\_truth}, i} \right|}{v_{\text{ground\_truth}, i}} \times 100\%, \quad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (v_{\text{calculated}, i} - v_{\text{ground\_truth}, i})^2}$$
- **Target Performance:** $\text{MAE} \le 2.5\text{ km/h}$ and $\text{MRE} \le 3.5\%$.

### 3.3 License Plate Recognition Test Set
- **Dataset:** **UFPR-ALPR Dataset** (Federal University of Paraná)
- **Scope:** 4,500 fully annotated frames across 150 vehicle tracks recorded in real-world traffic with 30 consecutive frames per vehicle, enabling thorough benchmark evaluation of our temporal multi-frame candidate consensus locking mechanism.
- **License:** Public Academic & Research License.
- **Metrics:**
  - **Character Recognition Accuracy (CRA %):** Ratio of correctly predicted alphanumeric characters to total characters.
  - **Full Plate Accuracy (FPA %):** Ratio of completely matched plates (100% exact match) to total evaluated vehicle tracks. Target: $\text{FPA} \ge 90\%$.

---

## 4. Hardware Independence & CPU Execution Strategy

GitHub Actions CI runners (`ubuntu-latest`) and standard university evaluation computers do not possess dedicated NVIDIA GPUs. SMART TRAFFIC PRO is designed with a **CPU-First Architecture**:

1. **Inference Engine:** 
   - YOLO11 vehicle detector runs via PyTorch's native CPU multi-threaded backend (`device="cpu"`).
   - FastALPR operates using ONNX Runtime with `CPUExecutionProvider` configured, eliminating CUDA/cuDNN installation requirements.
2. **Video Encoding:**
   - Video writing and stream encoding utilize OpenCV software FourCC codecs (`mp4v` or CPU-based `libx264`).
   - Hardware-accelerated NVIDIA NVENC (`h264_nvenc`) and CUDA execution (`--device 0`) are treated as **strictly optional performance accelerators**. The pipeline queries hardware capabilities at initialization and seamlessly defaults to CPU without throwing runtime errors.
3. **CI/CD Quality Gate:**
   - The GitHub Actions workflow (`.github/workflows/ci.yml`) runs unit and integration tests inside a standard headless Ubuntu runner without GPU, verifying PEP8 standards (Flake8) and test coverage ($\ge 60\%$).

---

## 5. Privacy and Data Protection Protocol (KVKK / GDPR Compliance)

Vehicle license plate numbers are legally recognized as personally identifiable data under the Turkish Personal Data Protection Law (KVKK Law No. 6698) and the European General Data Protection Regulation (GDPR).

1. **Demonstration & Evaluation Data:**
   - All live demonstrations, defense presentations, and public repositories will strictly utilize videos from public research datasets (BrnoCompSpeed, UA-DETRAC, UFPR-ALPR) that were released with explicit consent for academic evaluation.
2. **Anonymization of Real-World Recordings:**
   - If local highway or street footage is captured for testing or calibration, an automated privacy filter is applied:
     - All detected license plates belonging to vehicles *not* involved in speeding infractions are automatically blurred using Gaussian filters (`cv2.GaussianBlur(crop, (25, 25), 30)`).
     - Passenger and driver cabin windows are obscured via pixelation downsampling to prevent facial identification.
3. **Data Retention & Disposal Policy:**
   - Raw, unedited video footage is stored exclusively on local encrypted drives and retained for a maximum duration of **7 calendar days** solely for algorithmic verification.
   - A scheduled cleanup script automatically purges raw video files older than 7 days.
   - Only cryptographic, cropped evidence snapshots of confirmed speeding infractions are retained in `output/ihlaller/` for academic evaluation. Upon project completion and defense examination, all private personal data will be securely shredded.

---

## 6. Concrete Web Interface & System Layers

### 6.1 Web Interface Technology Stack
- **Web Frontend (ArgusVision Web SOC):** Built with **HTML5, Vanilla CSS3 (Glassmorphism & Cyber-Command dark palette), and Vanilla ES6+ JavaScript**. Employs standard Fetch API calls to poll `/api/stats` and `/api/records`, an MJPEG `<img>` stream player for `/video_feed`, and interactive HTML5 modals for manual plate editing.
- **Administrative Portal:** A **Streamlit** dashboard (`app.py`) providing tabular filtering, CSV/database download buttons, and lane polygon visualization.
- **Backend / REST API:** **Python Flask** (`server.py`) with threaded request handling, CORS support, and JSON response endpoints.

### 6.2 The Four Concrete System Layers
1. **Presentation Layer:**
   - Web SOC Dashboard (`web/index.html`, `web/css/`, `web/js/`)
   - Streamlit Analytical Portal (`app.py`)
   - Command Line Interface (`main.py`, `zone_calibrator.py`)
2. **API & Service Orchestration Layer:**
   - Flask REST Controller (`server.py`)
   - Threaded Stream Controller (`StreamController`)
   - Session & File Upload Handler
3. **Core AI & Processing Pipeline Layer:**
   - Orchestration Pipeline: `TrafficPipeline` (`src/pipeline.py`)
   - Detection & Tracking: `VehicleDetector` (`src/detector.py`)
   - Velocity Calculation: `SpeedEstimator` (`src/speed_estimator.py`)
   - License Plate Recognition: `PlateRecognizer` (`src/plate_recognizer.py`)
   - Violation Management: `ViolationManager` (`src/violation_manager.py`)
   - Visual Rendering & HUD: `TrafficVisualizer` (`src/visualizer.py`)
4. **Data Persistence & Storage Layer:**
   - Relational Database: SQLite3 (`output/traffic_surveillance.db`)
   - Tabular Reporting: CSV Export Engine (`output/arac_kayitlari.csv`)
   - File Evidence Store: JPEG Snapshot Directory (`output/ihlaller/*.jpg`)
   - Configuration Store: YAML Specification (`config/default_config.yaml`)
