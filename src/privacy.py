"""
KVKK / GDPR Gizlilik Modülü (FR-07, NFR-5)

İhlal yapmayan araçların plakaları hem ekranda gösterilen hem de diske kaydedilen
karelerde bulanıklaştırılır. İhlal yapan aracın plakası kanıt olduğu için açık kalır.

Plaka konumu, OCR sırasında bulunan plaka kutusunun araç kutusuna göre GÖRELİ
konumundan tahmin edilir (araç hareket etse de plaka aynı göreli yerde kalır).
Plaka hiç görülmediyse araç kutusunun alt-orta bölgesi güvenli varsayılan olarak
bulanıklaştırılır (önce gizlilik).
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

# Plaka hiç görülmemişse kullanılan göreli bölge (araç kutusuna göre)
DEFAULT_REL = (0.22, 0.62, 0.78, 0.98)


class PlatePrivacy:
    def __init__(self, enabled: bool = True, margin: float = 0.5):
        self.enabled = enabled
        self.margin = margin
        self.rel_boxes: Dict[int, Tuple[float, float, float, float]] = {}

    def observe_plate(self, track_id: int, vehicle_box: Box, plate_box: Optional[Box]):
        """OCR'ın bulduğu plaka kutusunu araç kutusuna göre göreli olarak saklar."""
        if plate_box is None:
            return
        x1, y1, x2, y2 = vehicle_box
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        px1, py1, px2, py2 = plate_box
        rel = ((px1 - x1) / w, (py1 - y1) / h, (px2 - x1) / w, (py2 - y1) / h)
        if all(-0.2 <= v <= 1.2 for v in rel):
            self.rel_boxes[track_id] = rel

    def region_for(self, track_id: int, vehicle_box: Box, frame_shape) -> Box:
        x1, y1, x2, y2 = vehicle_box
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        rx1, ry1, rx2, ry2 = self.rel_boxes.get(track_id, DEFAULT_REL)
        bx1, by1, bx2, by2 = x1 + rx1 * w, y1 + ry1 * h, x1 + rx2 * w, y1 + ry2 * h
        mw, mh = (bx2 - bx1) * self.margin, (by2 - by1) * self.margin
        H, W = frame_shape[:2]
        return (int(max(0, bx1 - mw)), int(max(0, by1 - mh)),
                int(min(W, bx2 + mw)), int(min(H, by2 + mh)))

    @staticmethod
    def blur(frame: np.ndarray, region: Box) -> None:
        """Bölgeyi yerinde pikselleştirir (geri döndürülemez anonimleştirme)."""
        x1, y1, x2, y2 = region
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def forget(self, track_id: int):
        self.rel_boxes.pop(track_id, None)
