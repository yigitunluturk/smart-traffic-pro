# AI Model Evaluation Plan & Baseline Comparison
**SEN499 Graduation Project · Section 3.3 Artificial Intelligence Compliance**  
**Project:** SMART TRAFFIC PRO — Speed Estimation & ALPR

> **Status note:** The repository at the `v0-baseline` tag uses the stock pre-trained
> `yolo11m.pt` and standard FastALPR. **This document contains no measured results.**
> All numbers below are *targets* taken from the RSA document. Results will be added
> only after `scripts/eval_speed.py`, `scripts/eval_alpr.py` and the detector
> evaluation have been run on the datasets described here.

---

## 1. Baseline Comparison (as in the RSA, Section 4)

| Task | Baseline(s) | Proposed | Metric / target |
| :--- | :--- | :--- | :--- |
| Vehicle detection | Stock pre-trained YOLO11m (`v0-baseline`); stock YOLOv8n as a lightweight reference | YOLO11m fine-tuned on UA-DETRAC | mAP@50 on the held-out UA-DETRAC test split; target **+8 points** over stock YOLO11m |
| Plate recognition | (a) EasyOCR on plate crops; (b) FastALPR single frame | FastALPR + temporal consensus buffer + Turkish syntax validation | Full plate accuracy (FPA) on the Turkish test set; target **FPA ≥ 90%** |
| Speed estimation | Naive pixel-displacement speed (no perspective calibration) | Perspective calibration with d_real (`v0-baseline`) | MAE on BrnoCompSpeed; target **≤ 2.0 km/h** |

---

## 2. Datasets

- **UA-DETRAC** — detector fine-tuning (GPU on Google Colab) and detection evaluation.
  A fixed train/validation/test split is stored in the dataset configuration file.
- **BrnoCompSpeed** — speed evaluation against laser-gate ground-truth speeds.
- **Turkish plate test set** — plate recognition evaluation. Source, size, license and
  KVKK handling are defined in the updated RSA document. Plate images are personal data
  and are **not** committed to this repository.
- **UFPR-ALPR is not used**: it contains Brazilian plates, which differ from the Turkish format.

---

## 3. Target Plate Format (Turkey)

Province code `01`–`81`, one to three letters (Q, W, X not used) and two to four digits,
e.g. `34 ABC 123`. Invalid OCR candidates are flagged as low confidence and can be
corrected manually in the Streamlit dashboard (FR-09).

---

## 4. Metrics

- **mAP@50** — standard Ultralytics validation on the UA-DETRAC test split.
- **Full plate accuracy (FPA)** — share of test plates where every character is correct.
- **Speed MAE**

```math
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| v_{\text{estimated}, i} - v_{\text{ground-truth}, i} \right|
```

- **Throughput** — dashboard FPS counter on a 1080p input at `imgsz=640` (target ≥ 10 FPS on CPU, NFR-3).

---

## 5. Results

*To be filled in only with numbers produced by the scripts above.*
