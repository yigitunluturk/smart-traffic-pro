import argparse
import json
import cv2
import numpy as np


class ZoneCalibrator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.points = []
        self.all_polygons = []
        self.frame = None
        self.display_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            print(f"👉 Nokta eklendi: [{x}, {y}] (Toplam: {len(self.points)})")
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"↩️ Son nokta geri alındı: {removed}")
                self._redraw()

    def _redraw(self):
        self.display_frame = self.frame.copy()
        
        for idx, poly in enumerate(self.all_polygons):
            pts = np.array(poly["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.display_frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(self.display_frame, poly.get("name", f"Poly_{idx}"),
                        tuple(poly["points"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for idx, pt in enumerate(self.points):
            cv2.circle(self.display_frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(self.display_frame, str(idx + 1), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if len(self.points) > 1:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.display_frame, [pts], False, (0, 255, 255), 2)

        cv2.putText(self.display_frame, "Sol Tik: Nokta Ekle | Sag Tik: Geri Al | 's': Poligonu Kaydet | 'c': Temizle | 'q': Cikis",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Zone Calibrator", self.display_frame)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"❌ Video karesi okunamadı: {self.video_path}")
            return

        self.frame = frame
        self.display_frame = frame.copy()

        cv2.namedWindow("Zone Calibrator")
        cv2.setMouseCallback("Zone Calibrator", self.mouse_callback)
        self._redraw()

        print("\n" + "=" * 60)
        print("📐 BÖLGE KALİBRASYON ARACI BAŞLATILDI")
        print("=" * 60)
        print("Sol Tık    : Nokta Ekle")
        print("Sağ Tık   : Son Noktayı Sil")
        print("'s' Tuşu   : Aktif Noktaları Poligon Olarak Kaydet")
        print("'c' Tuşu   : Aktif Noktaları Temizle")
        print("'q' Tuşu   : Çıkış ve Koordinatları JSON Olarak Yazdır")
        print("=" * 60 + "\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if len(self.points) >= 3:
                    name = f"Zone_{len(self.all_polygons) + 1}"
                    self.all_polygons.append({"name": name, "points": self.points.copy()})
                    print(f"✅ Poligon kaydedildi: {name} -> {self.points}")
                    self.points.clear()
                    self._redraw()
                else:
                    print("⚠️ Bir poligon oluşturmak için en az 3 nokta seçmelisiniz.")
            elif key == ord("c"):
                self.points.clear()
                print("🧹 Aktif noktalar temizlendi.")
                self._redraw()

        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("📋 ÇIKTI KOORDİNATLARI (config/default_config.yaml içine ekleyebilirsiniz):")
        print("=" * 60)
        print(json.dumps(self.all_polygons, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="video.mp4", help="Video yolu")
    args = parser.parse_args()
    calibrator = ZoneCalibrator(args.input)
    calibrator.run()
