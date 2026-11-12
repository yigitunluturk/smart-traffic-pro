# Artificial Intelligence Tool Usage Declaration
**SEN499 Graduation Project · Section 8 Academic Integrity Compliance**  
**Student Name:** *Yiğit ÜNLÜTÜRK* &nbsp;|&nbsp; **Student ID:** *210129060*  
**Academic Year:** 2026–2027 Fall Semester

---

## 1. Compliance Statement
In accordance with Section 8 (*Yapay Zekâ Araçlarının Kullanımı ve Akademik Dürüstlük*) of the SEN499 Graduation Project Guidelines, I hereby transparently declare the artificial intelligence tools utilized during the research, software development, testing, and documentation phases of this graduation project.

I acknowledge that I am fully responsible for every line of source code, architectural decision, and documentation submitted. I am prepared to explain, defend, and perform live code modifications upon request during the interim and final defense presentations.

---

## 2. Inventory of Utilized AI Tools

| AI Tool / Assistant | Purpose & Application Area | Contribution Scope |
| :--- | :--- | :--- |
| **Google Gemini / Antigravity IDE Agent** | Software architecture scaffolding, code refactoring, test suite generation, and documentation drafting. | Assisting in modular Python class design, Pytest fixture structuring, Docker containerization, and English documentation formatting. |
| **Ultralytics YOLO11m Pre-trained Weights** | Deep learning computer vision model for multi-class vehicular object detection. | Pre-trained feature extractor and bounding-box regressor on the COCO dataset. |
| **fast-alpr (ONNX Models)** | Deep learning inference for license plate detection (YOLOv9-t) and OCR character recognition (CCT-S-v2). | Pre-trained ONNX neural network weights for localized license plate reading. |

---

## 3. Human Engineering Contributions & Original Work
While pre-trained neural networks and AI coding assistants were leveraged for efficiency, all core intellectual engineering decisions were uniquely designed and implemented for this project:
1. **Mathematical Speed Estimation Algorithm:** Formulation of the framerate-timeline transit equation independent of processing speed, with resolution auto-scaling.
2. **Multi-Frame OCR Candidate Locking Mechanism:** Original temporal buffer algorithm preventing plate oscillation by collecting $N$ valid candidate readings.
3. **Interactive Zone Calibration Architecture:** Custom OpenCV mouse callback system allowing real-time calibration for arbitrary camera viewpoints.
4. **End-to-End System Pipeline Integration:** Harmonious multi-threaded architectural coordination connecting detection, tracking, speed, ALPR, SQLite persistence, and UI.

---
