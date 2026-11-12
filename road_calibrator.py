"""
Yol Düzlemi (Homografi) Kalibrasyon Aracı — doğru hız ölçümü için

Kullanım:
    python road_calibrator.py --input output/uploads/video.mp4 --width 3.5 --length 24

Adımlar:
  1. Videonun İLK karesi açılır (hız hesabı bu kareyi referans alır; kamera kayması otomatik telafi edilir).
  2. Yol üzerinde gerçek ölçüleri bilinen bir dikdörtgenin 4 köşesine SIRAYLA tıklayın:
        1: yakın-sol   2: yakın-sağ   3: uzak-sağ   4: uzak-sol
     Örnek: iki şerit çizgisi (aralık = şerit genişliği, TR'de tipik 3.5 m) ve
     kesik çizgilerin başlangıçları (uzunluk = periyot sayısı x periyot).
     Ölçüleri sahada veya Google Earth cetveliyle doğrulamanız önerilir.
  3. Sağ tık: son noktayı geri al.   's': kaydet.   'q': çıkış.

Çıktı: config/calibrations/<video_adı>.json  (pipeline bu dosyayı otomatik yükler)
"""
import argparse
import json
import os

import cv2
import numpy as np

BASE_W, BASE_H = 1280, 720


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Video dosyası")
    ap.add_argument("--width", type=float, default=3.5, help="Dikdörtgenin genişliği (m), örn. şerit genişliği")
    ap.add_argument("--length", type=float, default=24.0, help="Dikdörtgenin uzunluğu (m), yol boyunca")
    ap.add_argument("--out-dir", default="config/calibrations")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Video okunamadı: {args.input}")

    base = cv2.resize(frame, (BASE_W, BASE_H))
    pts = []
    labels = ["1 yakin-sol", "2 yakin-sag", "3 uzak-sag", "4 uzak-sol"]

    def redraw():
        img = base.copy()
        for i, p in enumerate(pts):
            cv2.circle(img, p, 5, (0, 0, 255), -1)
            cv2.putText(img, labels[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) == 4, (0, 255, 0), 2)
        if len(pts) == 4:
            # Doğrulama ızgarası: her 1 m enine, her 3 m boyuna
            img_pts = np.float32(pts)
            wld = np.float32([[0, 0], [args.width, 0], [args.width, args.length], [0, args.length]])
            Hinv = cv2.getPerspectiveTransform(wld, img_pts)
            for y in np.arange(-6, args.length + 12, 3.0):
                line = cv2.perspectiveTransform(np.float32([[[-args.width, y], [2 * args.width, y]]]), Hinv)[0]
                cv2.line(img, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 200, 0), 1)
            cv2.putText(img, "Izgara cizgileri yol boyunca 3 m aralikli olmali. 's' = kaydet",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(img, f"Tiklayin: {labels[len(pts)]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Yol Kalibrasyonu", img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            redraw()

    cv2.namedWindow("Yol Kalibrasyonu")
    cv2.setMouseCallback("Yol Kalibrasyonu", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and len(pts) == 4:
            stem = os.path.splitext(os.path.basename(args.input))[0]
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, f"{stem}.json")
            data = {
                "base_resolution": [BASE_W, BASE_H],
                "image_points": [list(map(float, p)) for p in pts],
                "world_points": [[0.0, 0.0], [args.width, 0.0], [args.width, args.length], [0.0, args.length]],
                "measure_y_range": [0.45, 0.985],
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Kaydedildi: {out}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
