"""
FR-03: YOLO11m'i UA-DETRAC üzerinde ince ayar (Google Colab GPU'da çalıştırılır)

Colab:
  !git clone <repo> && cd smart-traffic-pro && pip install -r requirements.txt
  !python scripts/prepare_ua_detrac.py --root /content/UA-DETRAC --out data/ua_detrac --every 10
  !python scripts/train_detector.py --data data/ua_detrac/ua_detrac.yaml --epochs 30 --device 0

Çıktı: models/yolo11m_uadetrac.pt  (config/default_config.yaml -> model.vehicle_model)
Çıkarım (inference) yine CPU'da çalışır (NFR-4).
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    res = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      device=args.device, seed=args.seed, deterministic=True,
                      project="runs/detect", name="yolo11m_uadetrac", exist_ok=True)
    best = os.path.join(str(res.save_dir), "weights", "best.pt")
    os.makedirs("models", exist_ok=True)
    shutil.copy(best, "models/yolo11m_uadetrac.pt")
    print("✅ models/yolo11m_uadetrac.pt kaydedildi.")


if __name__ == "__main__":
    main()
