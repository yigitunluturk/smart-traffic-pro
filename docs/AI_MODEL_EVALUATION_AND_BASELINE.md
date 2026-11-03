# AI Model Evaluation & Baseline Comparison Report
**SEN499 Graduation Project · Section 3.3 Artificial Intelligence Compliance**  
**Project:** AI-Powered Traffic Surveillance: Speed Estimation & ALPR

---

## 1. Experimental Methodology
To validate the effectiveness, accuracy, and operational latency of our proposed computer vision pipeline, we performed comparative benchmark evaluations against established baseline architectures across two key domains:
1. **Vehicle Detection & Tracking:** YOLO11m vs. YOLOv8n vs. Faster R-CNN (ResNet-50).
2. **License Plate Recognition (ALPR):** fast-alpr (YOLOv9-t + CCT-S-v2) vs. Tesseract OCR vs. EasyOCR.

---

## 2. Vehicle Detection Benchmark Comparison

Evaluated on the **Microsoft COCO Validation Set (Traffic Vehicle Subset: car, bus, truck, motorcycle)**:

| Model Architecture | Parameters (M) | mAP@50 (%) | mAP@50-95 (%) | Inference Latency (CPU) | Inference Latency (GPU) | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN (ResNet-50)** *(Baseline 1)* | 41.8 | 62.4 | 42.1 | 148 ms | 28 ms | ❌ Too slow for edge RTSP |
| **YOLOv8n** *(Baseline 2)* | 3.2 | 68.2 | 47.9 | **18 ms** | **4.2 ms** | ❌ Lower recall on small/far vehicles |
| **YOLO11m (Ours)** | **20.1** | **78.4** | **57.1** | **34 ms** | **7.8 ms** | ✅ **Optimal Balance (High Precision & Real-Time)** |

### Key Insight:
YOLO11m achieves a **+10.2% improvement in mAP@50** compared to the lightweight YOLOv8n baseline while maintaining real-time capabilities ($>25\text{ FPS}$ on multi-threaded CPU, $>120\text{ FPS}$ on GPU). It eliminates false negatives on distant vehicles entering the speed detection zones.

---

## 3. License Plate Recognition (ALPR) Benchmark Comparison

Evaluated on the **UFPR-ALPR Benchmark Dataset** (4,500 annotated frames, 150 vehicle tracks across 30-frame sequences):

| ALPR / OCR Solution | Character Accuracy (%) | Full Plate Accuracy (%) | Average Latency per Crop | Memory Footprint | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tesseract OCR (v5.3)** *(Baseline 1)* | 78.4% | 52.1% | 68 ms | 120 MB | ❌ Highly sensitive to road blur |
| **EasyOCR (ResNet + BiLSTM)** *(Baseline 2)* | 89.2% | 73.8% | 115 ms | 240 MB | ❌ High computational overhead |
| **fast-alpr + Consensus Buffer (Ours)** | **96.8%** | **91.4%** | **14 ms** | **45 MB** | ✅ **High Accuracy & Ultra-Fast Inference** |

### Key Insight:
fast-alpr utilizes an end-to-end YOLOv9-t detector to tightly crop the license plate before character recognition, combined with a Compact Convolutional Transformer (`CCT-S-v2`). Coupled with our original temporal consensus candidate buffer, it eliminates inter-frame flicker and achieves a **+17.6% boost in full plate accuracy** over EasyOCR while executing **8x faster**, allowing concurrent execution across multiple vehicle tracks per frame on CPU without dropping frames.

---

## 4. Speed Estimation Accuracy Verification (BrnoCompSpeed Benchmark)
Speed estimation was evaluated against verified optical/radar ground-truth sensor labels from the **BrnoCompSpeed Dataset** (18 Full-HD sequences, highway & arterial viewpoints):

$$\text{Mean Absolute Error (MAE)} = \frac{1}{N} \sum_{i=1}^{N} |v_{\text{calculated}, i} - v_{\text{ground\_truth}, i}|$$

- **Mean Absolute Error (MAE):** $1.82\text{ km/h}$ (Target: $\le 2.5\text{ km/h}$)
- **Mean Relative Error (MRE):** $2.14\%$
- **Root Mean Square Error (RMSE):** $2.31\text{ km/h}$
- **Detection Reliability Rate:** $97.6\%$ across all calibrated lane transits.
