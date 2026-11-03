from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .speed_estimator import SpeedZoneConfig, VehicleSpeedInfo
from .plate_recognizer import PlateDetectionResult
from .detector import VehicleDetection


class FrameVisualizer:
    """
    Üst düzey Siber Trafik ve Otonom Araç HUD Görselleştirme Katmanı.
    Özellikler:
    - Yarı saydam siber kutular ve neon köşe vizörleri (Corner Reticles)
    - Araç hareket yörüngesi kuyrukları (Fading Motion Trails)
    - Gerçekçi Avrupa/Türk tipi plaka grafiği (Mavi şeritli plaka rozeti)
    - Hız durumuna göre dinamik parlayan hız hapları (Speed Pills)
    - Yarı saydam akıllı şerit poligonları ve yön okları
    - Profesyonel Siber Komuta Merkezi (SOC) HUD Paneli
    """
    def __init__(
        self,
        hud_enabled: bool = True,
        box_thickness: Optional[int] = None,
        font_scale: Optional[float] = None,
        speed_limit: int = 80
    ):
        self.hud_enabled = hud_enabled
        self.user_box_thickness = box_thickness
        self.user_font_scale = font_scale
        self.speed_limit = speed_limit

        # Araç hareket yörünge geçmişi: {track_id: deque([(cx, cy), ...], maxlen=20)}
        self.trails: Dict[int, deque] = {}

        # Siber Renk Paleti (BGR)
        self.COLOR_CYAN = (254, 242, 0)         # Neon Camgöbeği (Cyan)
        self.COLOR_EMERALD = (80, 210, 110)     # Neon Zümrüt Yeşili (Normal Hız)
        self.COLOR_CRIMSON = (45, 45, 245)      # Neon Kırmızı / Kızıl (Hız İhlali)
        self.COLOR_AMBER = (0, 190, 255)        # Parlak Kehribar Sarı
        self.COLOR_PURPLE = (220, 70, 160)      # Neon Mor / Macenta
        self.COLOR_DARK_SLATE = (18, 14, 10)    # Koyu Grafit Siyahı
        self.COLOR_WHITE = (255, 255, 255)

    def _get_scale(self, frame_h: int) -> Tuple[float, int, int]:
        """Çözünürlüğe (720p, 1080p, 4K) göre dinamik ölçek çarpanı."""
        factor = max(0.65, frame_h / 1080.0)
        f_scale = self.user_font_scale if self.user_font_scale is not None else (0.60 * factor)
        b_thick = self.user_box_thickness if self.user_box_thickness is not None else max(2, int(2 * factor))
        f_thick = max(1, int(1.8 * factor))
        return f_scale, b_thick, f_thick

    def draw_zones(self, frame: np.ndarray, zones: List[SpeedZoneConfig]):
        """Şerit bölgelerini şeffaf neon dolgu ve yön göstergeleriyle çizer."""
        h, w = frame.shape[:2]
        factor = max(0.65, h / 1080.0)
        f_scale, b_thick, f_thick = self._get_scale(h)

        overlay = frame.copy()
        alpha = 0.22

        for z in zones:
            # 1. Giriş Bölgesi (Neon Sarı/Kehribar)
            if len(z.entry_polygon) > 0:
                cv2.fillPoly(overlay, [z.entry_polygon], z.color_entry)
                cv2.polylines(frame, [z.entry_polygon], True, z.color_entry, b_thick, cv2.LINE_AA)
                
                # Giriş Etiketi
                pt = tuple(z.entry_polygon[0][0])
                label = f"▼ {z.name} [ENTRY]" if z.direction == "downward" else f"▲ {z.name} [ENTRY]"
                cv2.putText(frame, label, (pt[0], max(24, pt[1] - int(10 * factor))),
                            cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.85, z.color_entry, f_thick, cv2.LINE_AA)

            # 2. Çıkış Bölgesi (Kırmızı/Bitiş)
            if len(z.exit_polygon) > 0:
                cv2.fillPoly(overlay, [z.exit_polygon], z.color_exit)
                cv2.polylines(frame, [z.exit_polygon], True, z.color_exit, b_thick, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_plate_ocr_band(self, frame: np.ndarray, top_ratio: float, bottom_ratio: float):
        """Plaka okuma bandını şık teknolojik kesikli/siber çizgilerle çizer."""
        h, w = frame.shape[:2]
        y_top = int(h * top_ratio)
        y_bottom = int(h * bottom_ratio)
        f_scale, b_thick, f_thick = self._get_scale(h)

        # Üst ve Alt sınır çizgileri
        cv2.line(frame, (0, y_top), (w, y_top), self.COLOR_PURPLE, max(1, b_thick // 2), cv2.LINE_AA)
        cv2.line(frame, (0, y_bottom), (w, y_bottom), self.COLOR_PURPLE, max(1, b_thick // 2), cv2.LINE_AA)

        # Şık sol köşe rozeti
        badge_text = "⚡ ALPR OCR DETECTION ZONE"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.7, 1)
        pad = int(8 * (h / 1080.0))
        cv2.rectangle(frame, (20, y_top - th - pad * 2), (20 + tw + pad * 2, y_top), self.COLOR_DARK_SLATE, -1)
        cv2.rectangle(frame, (20, y_top - th - pad * 2), (20 + tw + pad * 2, y_top), self.COLOR_PURPLE, 1)
        cv2.putText(frame, badge_text, (20 + pad, y_top - pad),
                    cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.7, self.COLOR_PURPLE, 1, cv2.LINE_AA)

    def draw_vehicle(
        self,
        frame: np.ndarray,
        det: VehicleDetection,
        speed_info: VehicleSpeedInfo,
        plate_res: PlateDetectionResult
    ):
        """Tek bir araç için fütüristik köşe vizörü, hız rozeti ve gerçekçi plaka çizer."""
        x1, y1, x2, y2 = det.box
        cx, cy = det.center
        h, w = frame.shape[:2]
        factor = max(0.65, h / 1080.0)
        f_scale, b_thick, f_thick = self._get_scale(h)

        is_viol = speed_info.is_speeding
        main_color = self.COLOR_CRIMSON if is_viol else self.COLOR_EMERALD

        # 1. Araç Hareket Kuyruğu (Motion Trajectory Trail)
        if det.track_id not in self.trails:
            self.trails[det.track_id] = deque(maxlen=16)
        self.trails[det.track_id].append((cx, cy))

        trail_pts = list(self.trails[det.track_id])
        for i in range(1, len(trail_pts)):
            pt1 = trail_pts[i - 1]
            pt2 = trail_pts[i]
            thickness = max(1, int(factor * (i / len(trail_pts)) * 3))
            cv2.line(frame, pt1, pt2, main_color, thickness, cv2.LINE_AA)

        # 2. Yarı Saydam Araç Kutusu ve Köşe Vizörleri (Reticles)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), main_color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # İnce çevre kutusu
        cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 1, cv2.LINE_AA)

        # Parlayan 4 köşe L-vizörleri (Corner Brackets)
        corn_len = max(12, int(min(x2 - x1, y2 - y1) * 0.20))
        c_thick = max(2, int(b_thick * 1.5))

        # Sol-Üst
        cv2.line(frame, (x1, y1), (x1 + corn_len, y1), main_color, c_thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corn_len), main_color, c_thick)
        # Sağ-Üst
        cv2.line(frame, (x2, y1), (x2 - corn_len, y1), main_color, c_thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corn_len), main_color, c_thick)
        # Sol-Alt
        cv2.line(frame, (x1, y2), (x1 + corn_len, y2), main_color, c_thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corn_len), main_color, c_thick)
        # Sağ-Alt
        cv2.line(frame, (x2, y2), (x2 - corn_len, y2), main_color, c_thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corn_len), main_color, c_thick)

        # 3. Araç İçinde Tespit Edilen Plaka Kutusu (Varsa neon çerçeve)
        if plate_res.plate_box is not None:
            px1, py1, px2, py2 = plate_res.plate_box
            cv2.rectangle(frame, (px1, py1), (px2, py2), self.COLOR_PURPLE, 2, cv2.LINE_AA)

        # 4. Modern Bilgi Rozetleri (Yukarıdan aşağıya istiflenmiş şık kapsüller)
        badges = []

        # Rozet 1: Araç Kimlik ve Sınıf Başlığı
        badges.append({
            "type": "class",
            "text": f"#{det.track_id} {det.class_name.upper()}",
            "bg_color": (28, 24, 20),
            "text_color": (230, 240, 255),
            "border_color": (80, 80, 80)
        })

        # Rozet 2: Hız Göstergesi (Hız aşımı durumunda yanıp sönen kırmızı)
        if speed_info.speed_kmh is not None:
            if is_viol:
                speed_text = f"⚡ {speed_info.speed_kmh} km/h [VIOLATION!]"
                speed_bg = (30, 20, 220)       # Kırmızı
                speed_txt = (255, 255, 255)
                speed_border = (100, 100, 255)
            else:
                speed_text = f"⚡ {speed_info.speed_kmh} km/h"
                speed_bg = (25, 120, 45)       # Yeşil
                speed_txt = (240, 255, 240)
                speed_border = (80, 220, 110)

            badges.append({
                "type": "speed",
                "text": speed_text,
                "bg_color": speed_bg,
                "text_color": speed_txt,
                "border_color": speed_border
            })

        # Rozet 3: Plaka Kartı (Gerçekçi mavi bantlı TR/EU plaka stili)
        if plate_res.plate_text:
            conf_str = f" (%{int(plate_res.confidence * 100)})" if plate_res.confidence else ""
            status_str = " 🔒" if plate_res.is_locked else ""
            badges.append({
                "type": "plate",
                "text": f"{plate_res.plate_text}{conf_str}{status_str}",
                "plate_raw": plate_res.plate_text,
                "is_locked": plate_res.is_locked
            })

        # Rozetleri Çiz
        cur_y = y1 - int(8 * factor)
        pad_x = int(8 * factor)
        pad_y = int(5 * factor)

        for b in reversed(badges):
            if b["type"] == "plate":
                # Özel Gerçekçi Plaka Tasarımı (Beyaz Zemin, Mavi Sol Şerit, Siyah Karakterler)
                text = b["text"]
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.95, f_thick)
                
                box_w = tw + pad_x * 2 + int(24 * factor)
                box_h = th + pad_y * 2
                b_top = cur_y - box_h
                b_bot = cur_y

                if b_top < 0:
                    b_top = y2 + pad_y
                    b_bot = y2 + box_h + pad_y
                    cur_y = b_bot + pad_y

                # Beyaz Plaka Gövdesi
                cv2.rectangle(frame, (x1, b_top), (x1 + box_w, b_bot), (245, 245, 245), -1)
                # Dış Siyah Çerçeve
                cv2.rectangle(frame, (x1, b_top), (x1 + box_w, b_bot), (20, 20, 20), max(1, int(1.5 * factor)))

                # Mavi Sol Şerit (EU / TR Şeridi)
                blue_w = int(22 * factor)
                cv2.rectangle(frame, (x1, b_top), (x1 + blue_w, b_bot), (180, 80, 20), -1)
                cv2.putText(frame, "TR", (x1 + int(3 * factor), b_bot - int(5 * factor)),
                            cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # Plaka Metni (Koyu Siyah Embossed)
                cv2.putText(frame, text, (x1 + blue_w + pad_x, b_bot - pad_y),
                            cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.95, (10, 10, 10), f_thick, cv2.LINE_AA)
                cur_y = b_top - int(4 * factor)

            else:
                # Standart Siber Kapsül Rozet
                text = b["text"]
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.85, f_thick)
                box_w = tw + pad_x * 2
                box_h = th + pad_y * 2
                b_top = cur_y - box_h
                b_bot = cur_y

                if b_top < 0:
                    b_top = y2 + pad_y
                    b_bot = y2 + box_h + pad_y
                    cur_y = b_bot + pad_y

                # Arka Plan
                cv2.rectangle(frame, (x1, b_top), (x1 + box_w, b_bot), b["bg_color"], -1)
                # Çerçeve
                cv2.rectangle(frame, (x1, b_top), (x1 + box_w, b_bot), b["border_color"], 1, cv2.LINE_AA)
                # Metin
                cv2.putText(frame, text, (x1 + pad_x, b_bot - pad_y),
                            cv2.FONT_HERSHEY_DUPLEX, f_scale * 0.85, b["text_color"], f_thick, cv2.LINE_AA)
                cur_y = b_top - int(4 * factor)

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_no: int,
        fps: float,
        summary: Dict
    ):
        """Sol üst köşede siber komuta merkezi (SOC) gösterge paneli çizer."""
        if not self.hud_enabled:
            return

        h, w = frame.shape[:2]
        factor = max(0.65, h / 1080.0)
        panel_w = int(350 * factor)
        panel_h = int(175 * factor)
        pos_x = int(20 * factor)
        pos_y = int(20 * factor)

        # 1. Şeffaf Buzlu Cam Gövde
        overlay = frame.copy()
        cv2.rectangle(overlay, (pos_x, pos_y), (pos_x + panel_w, pos_y + panel_h), (12, 10, 8), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # 2. Neon Siber Çerçeve ve Köşeler
        cv2.rectangle(frame, (pos_x, pos_y), (pos_x + panel_w, pos_y + panel_h), (50, 65, 85), 1)
        
        # Üst Vurgu Çizgisi (Neon Camgöbeği)
        cv2.line(frame, (pos_x, pos_y), (pos_x + panel_w, pos_y), self.COLOR_CYAN, max(2, int(2 * factor)))

        f_scale = 0.50 * factor
        f_thick = max(1, int(factor))

        # Canlı Durum İkonu (Yeşil parlayan nokta)
        dot_x = pos_x + int(14 * factor)
        dot_y = pos_y + int(24 * factor)
        cv2.circle(frame, (dot_x, dot_y), int(5 * factor), (50, 220, 100), -1)

        # Başlık
        cv2.putText(frame, "ARGUS-VISION // TRAFFIC SOC", (dot_x + int(12 * factor), dot_y + int(4 * factor)),
                    cv2.FONT_HERSHEY_DUPLEX, f_scale * 1.05, (255, 255, 255), max(1, int(1.4 * factor)), cv2.LINE_AA)

        # Telemetri Satırları
        viol_count = summary.get("violations_count", 0)
        viol_color = self.COLOR_CRIMSON if viol_count > 0 else (180, 200, 220)

        lines = [
            (f"FPS: {fps:.1f}  |  FRAME: #{frame_no}", (160, 180, 200), f_scale * 0.9),
            (f"TRACKED VEHICLES: {summary.get('total_vehicles', 0)}", (240, 245, 255), f_scale * 0.92),
            (f"SPEED MEASURED  : {summary.get('speed_measured_count', 0)}", (100, 230, 140), f_scale * 0.92),
            (f"PLATES RECOGNIZED: {summary.get('plate_recognized_count', 0)}", (240, 160, 255), f_scale * 0.92),
            (f"SPEED ENFORCEMENT: {self.speed_limit} km/h", (255, 230, 80), f_scale * 0.92),
            (f"VIOLATIONS ALERT: {viol_count}", viol_color, f_scale * 1.0),
        ]

        start_y = pos_y + int(48 * factor)
        step_y = int(20 * factor)
        for text, col, scale in lines:
            cv2.putText(frame, text, (pos_x + int(14 * factor), start_y),
                        cv2.FONT_HERSHEY_DUPLEX, scale, col, f_thick, cv2.LINE_AA)
            start_y += step_y
