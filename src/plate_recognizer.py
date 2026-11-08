from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import numpy as np


@dataclass
class PlateCandidate:
    text: str
    confidence: float


@dataclass
class PlateTrackState:
    candidates: List[PlateCandidate] = field(default_factory=list)
    locked_text: Optional[str] = None
    locked_conf: Optional[float] = None
    running_text: Optional[str] = None
    last_plate_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in full frame
    is_locked: bool = False


@dataclass
class PlateDetectionResult:
    plate_text: Optional[str]
    confidence: Optional[float]
    is_locked: bool
    plate_box: Optional[Tuple[int, int, int, int]]
    running_text: Optional[str]


class PlateRecognizer:
    """
    fast-alpr ve OCR tabanlı araç plaka tespit ve doğrulama modülü.
    Aday toplama ve en yüksek güvenilirlik skoruna göre plaka kilitleme uygular.
    """
    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "cct-s-v2-global-model",
        zone_top_ratio: float = 0.40,
        zone_bottom_ratio: float = 0.95,
        min_ocr_conf: float = 0.40,
        n_valid_reads: int = 3,
        plate_format: str = "all"
    ):
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self.zone_top_ratio = zone_top_ratio
        self.zone_bottom_ratio = zone_bottom_ratio
        self.min_ocr_conf = min_ocr_conf
        self.n_valid_reads = n_valid_reads
        self.plate_format = plate_format

        # fast-alpr başlatma
        self.alpr = None
        self._init_alpr()

        # Track bazlı durum sözlüğü: {track_id: PlateTrackState}
        self.track_states: Dict[int, PlateTrackState] = {}

    def _init_alpr(self):
        try:
            import onnxruntime as ort
            try:
                ort.preload_dlls()
            except Exception:
                pass
            from fast_alpr import ALPR
            self.alpr = ALPR(
                detector_model=self.detector_model,
                ocr_model=self.ocr_model
            )
        except Exception as e:
            print(f"[PlateRecognizer] fast-alpr başlatılırken hata / uyarı: {e}")
            self.alpr = None

    @staticmethod
    def normalize_plate(text: str) -> str:
        """Karakterleri temizler ve büyük harfe çevirir."""
        return re.sub(r"[^A-Z0-9]", "", (text or "").upper())

    def is_valid_plate(self, text: str) -> bool:
        """Plaka format doğrulaması (TR, EU ve Genel)."""
        t = self.normalize_plate(text)
        if not (4 <= len(t) <= 10):
            return False
        
        has_digit = any(c.isdigit() for c in t)
        has_alpha = any(c.isalpha() for c in t)
        
        if not (has_digit and has_alpha):
            return False

        if self.plate_format == "tr":
            # Türkiye Plaka Formatları: 34ABC123, 06A1234, 35AB123 vb.
            # İlk 2 karakter il kodu (01-81)
            if not (t[:2].isdigit() and 1 <= int(t[:2]) <= 81):
                return False

        return True

    def process_vehicle(
        self,
        track_id: int,
        vehicle_box: Tuple[int, int, int, int],
        vehicle_center: Tuple[int, int],
        frame_clean: np.ndarray,
        frame_height: int,
        frame_no: int = 0
    ) -> PlateDetectionResult:
        """
        Aracın bölgesini ve plakasını analiz eder.
        Performans optimizasyonu: Boyut filtresi ve zaman aralığı (staggered sampling).
        """
        if track_id not in self.track_states:
            self.track_states[track_id] = PlateTrackState()

        st = self.track_states[track_id]
        cx, cy = vehicle_center
        x1, y1, x2, y2 = vehicle_box

        zone_y_top = int(frame_height * self.zone_top_ratio)
        zone_y_bottom = int(frame_height * self.zone_bottom_ratio)
        in_zone = zone_y_top <= cy <= zone_y_bottom

        # Kilitlenmişse veya zon dışındaysa OCR yapılmaz.
        # Çok küçük araçlarda (uzakta) plaka zaten okunamaz (min 60x40 px).
        # Her kare yerine per-arac 3 karede bir bakılır (CPU yükünü %66 düşürür).
        should_sample = ((frame_no + track_id) % 3 == 0) if frame_no > 0 else True
        w_box = x2 - x1
        h_box = y2 - y1

        if in_zone and not st.is_locked and self.alpr is not None and should_sample and (w_box >= 60 and h_box >= 40):
            # Araç görüntüsünü kırp
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(frame_clean.shape[1], x2), min(frame_clean.shape[0], y2)
            
            if (cx2 - cx1 > 50) and (cy2 - cy1 > 35):
                car_crop = frame_clean[cy1:cy2, cx1:cx2]
                best_read = self._predict_plate(car_crop)

                if best_read is not None:
                    ptext, pconf, (px1, py1, px2, py2) = best_read
                    st.last_plate_box = (cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2)
                    st.running_text = ptext

                    if pconf >= self.min_ocr_conf:
                        st.candidates.append(PlateCandidate(text=ptext, confidence=pconf))
                        if len(st.candidates) >= self.n_valid_reads:
                            # En yüksek güvenilirlikli okumayı kilitle
                            best_cand = max(st.candidates, key=lambda c: c.confidence)
                            st.locked_text = best_cand.text
                            st.locked_conf = round(best_cand.confidence, 2)
                            st.is_locked = True

        active_text = st.locked_text if st.locked_text else st.running_text
        active_conf = st.locked_conf if st.locked_conf else (st.candidates[-1].confidence if st.candidates else None)
        
        return PlateDetectionResult(
            plate_text=active_text,
            confidence=active_conf,
            is_locked=st.is_locked,
            plate_box=st.last_plate_box if in_zone else None,
            running_text=st.running_text
        )

    def _predict_plate(self, car_crop: np.ndarray) -> Optional[Tuple[str, float, Tuple[int, int, int, int]]]:
        if car_crop is None or car_crop.size == 0 or self.alpr is None:
            return None

        try:
            results = self.alpr.predict(car_crop)
            best = None
            for r in results:
                if r.ocr is None:
                    continue
                raw_text = r.ocr.text
                if not self.is_valid_plate(raw_text):
                    continue

                c = r.ocr.confidence
                conf = float(np.mean(c)) if isinstance(c, (list, tuple, np.ndarray)) and len(c) else float(c)

                bb = r.detection.bounding_box
                box = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))

                clean_text = self.normalize_plate(raw_text)
                if best is None or conf > best[1]:
                    best = (clean_text, conf, box)
            return best
        except Exception as e:
            return None
