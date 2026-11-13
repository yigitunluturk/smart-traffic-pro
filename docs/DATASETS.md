# Datasets, Licences and Personal-Data Handling

All demo and evaluation material comes from public datasets (advisor's condition, 22 Oct 2026).

| Purpose | Dataset | Source | Licence | Used by |
|---------|---------|--------|---------|---------|
| Detector fine-tuning and detection evaluation (FR-03) | **UA-DETRAC** (100 sequences, ~140k frames, 960×540, 25 fps) | https://sites.google.com/view/daweidu/projects/ua-detrac | CC BY-NC-SA (non-commercial research; exact version as stated on the dataset page) | `scripts/prepare_ua_detrac.py`, `train_detector.py`, `eval_detector.py` |
| Speed evaluation (NFR-1) | **BrnoCompSpeed** (Full-HD videos with LiDAR/optical-gate ground-truth speeds) | https://github.com/JakubSochor/BrnoCompSpeed (access by e-mail to the authors) | Research use, non-commercial | `scripts/eval_speed.py --brno` |
| Plate evaluation (NFR-2) | **Turkish plate test set** (see below) | Roboflow Universe "Turkish Number Plates" — https://universe.roboflow.com/plakatanima-vnt3k/turkish-number-plates | CC BY 4.0 | `scripts/eval_alpr.py` |

## UA-DETRAC split (reproducible)

`scripts/prepare_ua_detrac.py` splits the official training sequences **by sequence** (no frame leakage)
with `seed=0` into 85 % train / 15 % validation; the official test sequences form the test split.
Every 10th frame is used (consecutive frames are near-duplicates). Ignored regions are greyed out.
Classes are mapped to COCO ids so the pipeline configuration does not change:
`car→2, van→2, bus→5, others→7`. The exact sequence lists are written to `data/ua_detrac/split.json`.

## Turkish plate test set

* **Source:** plate images from the CC BY 4.0 Roboflow dataset above (the dataset provides plate
  bounding boxes; the plate *text* labels are added manually by the author).
* **Size:** **300 plates**, sampled once with a fixed seed and then frozen; the file list is versioned with
  `scripts/eval_alpr.py`. Plates that cannot be read reliably by eye are excluded before labelling.
* **Format:** `data/tr_plates/labels.csv` with columns `filename,plate` (one crop per row), or one folder per
  plate with several frames of the same vehicle for the temporal-voting test.
* **Ground truth:** every label must match the RSA §5.2 rule
  `^(0[1-9]|[1-7][0-9]|8[01])[A-PR-VYZ]{1,3}[0-9]{2,4}$`.
* UFPR-ALPR is **not** used (Brazilian plates).

## Personal data (KVKK Law No. 6698 / GDPR)

Licence plates are personal data. The project handles them as follows:

1. **Public data only.** Development, evaluation and the demo use only the public datasets above. No
   own recordings are made.
2. **Data minimisation in the test set.** Only plate crops are stored, not full scenes. The test set is kept
   locally under `data/` (git-ignored), is not redistributed, and is deleted after the project is graded.
3. **Blurring (FR-07).** In every displayed and stored frame, plates of non-violating vehicles are pixelated
   (`src/privacy.py`). Only the violating vehicle's plate stays visible in its evidence snapshot.
4. **Retention (NFR-5).** Plate text of non-violating transits is deleted after `privacy.retention_days`
   (default 30). This runs at start-up and can be run manually from screen S6. Violation evidence is kept.
5. **No enforcement.** This is an academic prototype. It makes no real enforcement decisions and issues no fines.
