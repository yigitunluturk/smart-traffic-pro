# Presentation & Defense Preparation Guide
**SEN499 Graduation Project · 10-Minute Presentation & Q&A Roadmap**  
**Project:** AI-Powered Traffic Surveillance: Speed Estimation & ALPR

> **Note:** This document is a preparation guide. Specific benchmark numbers  
> (mAP, MAE, FPA) will be filled in after evaluation scripts are run this semester.

---

## ⏱️ 10-Minute Presentation Timeline

| Time | Slide / Topic | Content & Key Talking Points |
| :--- | :--- | :--- |
| **0:00 - 1:30** | **1. Problem & Motivation** | • Traditional traffic enforcement uses expensive dual hardware (separate Doppler radar + dedicated ANPR gate).<br>• High cost, road-cutting physical sensors, and difficulty tracking dense multi-lane traffic simultaneously.<br>• **Our Solution:** A unified, single-camera vision pipeline that tracks, estimates speed, recognizes plates, and logs violations concurrently in real-time. |
| **1:30 - 3:00** | **2. System Architecture & Use Cases** | • Layered architecture: Presentation (Streamlit), Business Logic (`src/`), and Persistence (SQLite/CSV).<br>• Highlight the 6 Use Cases defined in the RSA document. |
| **3:00 - 5:30** | **3. Core Engineering Innovations** | • **Hardware-Independent Speed:** Speed uses video frame timeline $\Delta t = \frac{\Delta \text{Frames}}{\text{FPS}}$, making speed measurements exact regardless of inference lag.<br>• **Turkish Plate Validation:** Temporal consensus locking with TR syntax regex filtering.<br>• **Auto-Resolution Scaling:** Dynamic coordinate and font scaling across 720p, 1080p, and 4K. |
| **5:30 - 7:30** | **4. Live Demo Walkthrough** | • Launch Streamlit dashboard: `streamlit run app.py`.<br>• Select sample video.<br>• Show live vehicle tracking, speed tags, Turkish plate reading, and HUD stats.<br>• Show the **Violations Gallery** with photographic evidence and the **Database Tab** with plate search. |
| **7:30 - 9:00** | **5. AI Evaluation, Testing & CI/CD** | • Show baseline comparison results from `scripts/eval_speed.py` and `scripts/eval_alpr.py`.<br>• Present automated unit/integration tests running with `pytest`.<br>• Show GitHub Actions CI pipeline running automated linting and testing on every push. |
| **9:00 - 10:00** | **6. Conclusion & Future Work** | • Summary of engineering outcomes: Accurate speed estimation, Turkish ALPR support, fully containerized with Docker.<br>• Future enhancements: Fog/rain image de-weathering, night-vision infrared camera support. |

---

## 🎯 Anticipated Defense Questions & Model Answers

### Q1: "How do you ensure speed measurement accuracy if the processing computer slows down or drops frames?"
> **Answer:**  
> "Our speed estimation algorithm is completely decoupled from processing/hardware FPS. Instead of measuring time with `time.time()` (wall-clock time), we calculate time strictly from the video's intrinsic timeline:
> $$\Delta t = \frac{\text{Exit Frame} - \text{Entry Frame}}{\text{Video FPS}}$$
> Because video FPS is fixed by the recording camera hardware, even if the processing machine drops from 30 FPS to 5 FPS due to heavy load, the calculated speed remains mathematically exact."

---

### Q2: "Why did you choose YOLO11m instead of older models like YOLOv5 or YOLOv8?"
> **Answer:**  
> "YOLO11m is a stronger architecture than lightweight models such as YOLOv8n (C3k2 blocks, SPPF), which matters for distant or partially occluded vehicles. We do not claim this from the literature alone: stock YOLOv8n and stock YOLO11m are our baselines, and we compare them with our UA-DETRAC fine-tuned YOLO11m. The exact mAP figures will be reported after our evaluation scripts run on the standardized test split this semester."

---

### Q3: "License plate OCR can misread characters due to glare or motion blur. How does your system handle this?"
> **Answer:**  
> "We designed a multi-frame **Temporal Candidate Accumulation & Confidence Locking** algorithm (`src/plate_recognizer.py`). Rather than accepting a single-frame read, the system buffers candidate readings as the vehicle travels through the detection zone. Once 3 consistent readings are gathered (this semester, only candidates that pass the Turkish plate syntax validator count), the system locks the reading with the highest statistical OCR confidence to the vehicle's persistent `track_id`. Once locked, it stays attached to the vehicle even after it leaves the OCR zone."

---

### Q4: "Where is the surveillance data saved? How do you prevent data loss?"
> **Answer:**  
> "The system features a dual persistence layer:
> 1. An ACID-compliant **SQLite relational database** (`output/traffic_surveillance.db`) with dedicated tables for detection sessions, vehicle records, and speeding violations, indexed by license plate and timestamp.
> 2. Exportable **CSV files** and photographic JPEG evidence snapshots (`output/ihlaller/`) with bounding-box overlays for courtroom/administrative evidentiary presentation."

---

### Q5: "How does the project adhere to software engineering and testing standards?"
> **Answer:**  
> "The project follows a strictly modular 3-tier architecture with full separation of concerns. Automated unit and integration tests cover database queries, speed mathematics and pipeline integration; tests for the Turkish plate validator are added together with the validator this semester. All tests are integrated into our **GitHub Actions CI/CD** pipeline (`.github/workflows/ci.yml`), and the application is containerized with **Docker** and **Docker Compose** for zero-friction deployment on any operating system."
