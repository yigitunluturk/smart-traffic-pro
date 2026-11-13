"""
FR-03 / RSA Tablo 2: Araç tespiti karşılaştırması (mAP@50, UA-DETRAC test bölümü)

  python scripts/eval_detector.py --data data/ua_detrac/ua_detrac.yaml \
      --models yolo11m.pt yolov8n.pt models/yolo11m_uadetrac.pt

Yalnızca araç sınıfları (2 car, 5 bus, 7 truck) değerlendirilir. CPU'da çalışır.
Çıktı: output/eval/detector_comparison.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import write_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo11m.pt", "yolov8n.pt", "models/yolo11m_uadetrac.pt"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from ultralytics import YOLO
    metrics, per = {}, []
    for m in args.models:
        if not os.path.exists(m) and "/" in m:
            print(f"⚠️ {m} yok, atlanıyor.")
            continue
        r = YOLO(m).val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device,
                        classes=[2, 5, 7], verbose=False, plots=False)
        name = os.path.splitext(os.path.basename(m))[0]
        metrics[f"{name}_mAP50"] = round(float(r.box.map50) * 100, 2)
        metrics[f"{name}_mAP50_95"] = round(float(r.box.map) * 100, 2)
        per.append({"model": m, "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                    "speed_ms": r.speed})
    write_result("detector_comparison", metrics, vars(args), per,
                 notes="RSA hedefi: ince ayarlı YOLO11m, hazır YOLO11m'e göre +8 puan mAP@50.")


if __name__ == "__main__":
    main()
