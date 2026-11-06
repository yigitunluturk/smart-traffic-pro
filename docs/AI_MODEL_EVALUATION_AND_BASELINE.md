# AI Model Evaluation Planning & Target Baselines
**SEN499 Graduation Project · Section 3.3 Artificial Intelligence Compliance**  
**Project:** AI-Powered Traffic Surveillance: Speed Estimation & ALPR

> **Status Note (v0-baseline):** The current repository utilizes pre-trained baseline models (stock `yolo11m.pt` and standard FastALPR). The metrics, tables, and comparisons presented in this document represent the **academic targets and planned benchmarks** for this semester. Reproducible evaluation scripts and dataset parsing logic will be developed and committed during the semester to officially produce these results.

---

## 1. Planned Experimental Methodology
To validate the effectiveness, accuracy, and operational latency of our proposed computer vision pipeline against the `v0-baseline`, we will perform comparative benchmark evaluations against established baseline architectures across two key domains:
1. **Vehicle Detection & Tracking:** Pre-trained YOLO11m vs. Fine-tuned YOLO11m (Semester Target).
2. **License Plate Recognition (ALPR):** Standard OCR Baselines (Tesseract / EasyOCR) vs. fast-alpr with Temporal Consensus Buffer.

---

## 2. Vehicle Detection: Baseline vs. Semester Targets

Evaluations will be conducted using relevant subsets of traffic datasets (e.g., UA-DETRAC or COCO Traffic Subset) to measure improvements in multi-lane traffic detection.

| Model Architecture | Status | Target mAP@50 (%) | Target Inference Latency (CPU) |
| :--- | :---: | :---: | :---: |
| **YOLOv8n / Faster R-CNN** | *Literature Baselines* | ~ 60.0 - 68.0% | N/A |
| **YOLO11m (Pre-trained)** | *Current v0-baseline* | ~ 70.0% | ~ 35 ms |
| **YOLO11m (Fine-Tuned)** | **Semester Goal** | **≥ 78.0%** | **≤ 40 ms** |

### Semester Objective:
The primary goal is to fine-tune the existing pre-trained YOLO11m model on specific traffic datasets to achieve a minimum **78.0% mAP@50**. This aims to eliminate false negatives on distant/small vehicles entering the polygon speed detection zones while maintaining real-time CPU processing capabilities ($>25\text{ FPS}$).

---

## 3. ALPR: Baseline vs. Semester Targets

Evaluations will be engineered to test Turkish License Plate formats against standard benchmark subsets (such as UFPR-ALPR adapted concepts or custom annotated TR sets).

| ALPR / OCR Solution | Status | Target Full Plate Accuracy (%) | Target Latency per Crop |
| :--- | :---: | :---: | :---: |
| **Tesseract OCR (v5.3)** | *Baseline 1* | ~ 50.0 - 60.0% | ~ 65 ms |
| **EasyOCR (ResNet + BiLSTM)** | *Baseline 2* | ~ 70.0 - 75.0% | ~ 115 ms |
| **fast-alpr + TR Consensus Buffer**| **Semester Goal** | **≥ 90.0%** | **≤ 20 ms** |

### Semester Objective:
While the `v0-baseline` implements single-frame fast-alpr crops, the target for this semester is to integrate a temporal consensus buffer alongside a **Turkish Plate Syntax Regex (`01-81` province codes, etc.)**. The objective is to achieve a **Full Plate Accuracy of $\ge 90.0\%$**, severely outperforming standard EasyOCR latencies and stabilizing OCR flicker.

---

## 4. Speed Estimation Accuracy Verification (Planned)
Speed estimation will be rigorously evaluated against verified optical/radar ground-truth sensor labels using the **BrnoCompSpeed Dataset** (18 Full-HD sequences, highway & arterial viewpoints). 

A custom script (e.g., `scripts/eval_speed.py`) will be developed to calculate the error rates mathematically:

$$\text{Mean Absolute Error (MAE)} = \frac{1}{N} \sum_{i=1}^{N} |v_{\text{calculated}, i} - v_{\text{ground\_truth}, i}|$$

### Target Metrics (To Be Achieved):
- **Mean Absolute Error (MAE):** $\le 2.0\text{ km/h}$
- **Detection Reliability Rate:** $> 95.0\%$ across all calibrated lane transits.
