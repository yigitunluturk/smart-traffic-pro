from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


@dataclass
class SpeedZoneConfig:
    name: str
    direction: str  # "upward", "downward", "auto"
    raw_entry_polygon: List[List[int]]
    raw_exit_polygon: List[List[int]]
    entry_polygon: np.ndarray  # scaled (N, 1, 2)
    exit_polygon: np.ndarray   # scaled (N, 1, 2)
    calib_pixel_length: float
    real_distance_m: float
    color_entry: Tuple[int, int, int] = (0, 255, 255)
    color_exit: Tuple[int, int, int] = (0, 0, 255)


@dataclass
class VehicleSpeedInfo:
    track_id: int
    speed_kmh: Optional[int] = None
    is_calculated: bool = False
    lane_name: Optional[str] = None
    elapsed_time: Optional[float] = None
    entry_frame: Optional[int] = None
    exit_frame: Optional[int] = None
    is_speeding: bool = False


class SpeedEstimator:
    """
    Video FPS ve kalibre edilmiş yol bölgeleri üzerinden araç hızını hesaplayan modül.
    Farklı çözünürlükleri (720p, 1080p, 4K) otomatik ölçeklendirme desteğine sahiptir.
    """
    def __init__(
        self,
        zones_config: Optional[List[Dict[str, Any]]] = None,
        speed_limit: int = 80,
        base_resolution: Optional[Tuple[int, int]] = (1280, 720),
        min_valid_speed: int = 5,
        max_valid_speed: int = 240,
        band_top_ratio: float = 0.40,
        band_bottom_ratio: float = 0.95,
        corridor_real_distance_m: float = 24.0,
        use_corridor_mode: bool = False
    ):
        self.speed_limit = speed_limit
        self.base_resolution = base_resolution  # (base_w, base_h)
        self.min_valid_speed = min_valid_speed
        self.max_valid_speed = max_valid_speed
        self.raw_zones_config = zones_config or []
        self.current_resolution: Optional[Tuple[int, int]] = None
        self.zones: List[SpeedZoneConfig] = []
        
        self.band_top_ratio = band_top_ratio
        self.band_bottom_ratio = band_bottom_ratio
        self.corridor_real_distance_m = corridor_real_distance_m
        self.use_corridor_mode = use_corridor_mode

        if self.raw_zones_config:
            self._build_zones(self.base_resolution or (1280, 720))

        # Araç durum takibi: {track_id: {zone_name: {'entry_frame', 'entry_pos', ...}}}
        self.track_data: Dict[int, Dict[str, Any]] = {}
        # Nihai hesaplanan hız sonuçları: {track_id: VehicleSpeedInfo}
        self.results: Dict[int, VehicleSpeedInfo] = {}

    def _build_zones(self, target_resolution: Tuple[int, int]):
        self.current_resolution = target_resolution
        target_w, target_h = target_resolution
        base_w, base_h = self.base_resolution if self.base_resolution else (target_w, target_h)

        scale_x = target_w / max(1, base_w)
        scale_y = target_h / max(1, base_h)

        self.zones = []
        for z in self.raw_zones_config:
            raw_entry = z["entry_polygon"]
            raw_exit = z["exit_polygon"]

            # Otomatik çözünürlük ölçekleme
            scaled_entry = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in raw_entry]
            scaled_exit = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in raw_exit]

            entry_poly = np.array(scaled_entry, dtype=np.int32).reshape((-1, 1, 2))
            exit_poly = np.array(scaled_exit, dtype=np.int32).reshape((-1, 1, 2))
            
            calib_px = float(z.get("calib_pixel_length", 80.0)) * scale_y

            color_entry = tuple(z.get("color_entry", [0, 255, 255]))
            color_exit = tuple(z.get("color_exit", [0, 0, 255]))
            
            self.zones.append(
                SpeedZoneConfig(
                    name=z.get("name", "Zone"),
                    direction=z.get("direction", "auto"),
                    raw_entry_polygon=raw_entry,
                    raw_exit_polygon=raw_exit,
                    entry_polygon=entry_poly,
                    exit_polygon=exit_poly,
                    calib_pixel_length=calib_px,
                    real_distance_m=float(z.get("real_distance_m", 9.144)),
                    color_entry=color_entry,
                    color_exit=color_exit
                )
            )

    def adapt_to_frame(self, frame_w: int, frame_h: int):
        """Gelen video çözünürlüğüne göre poligonları otomatik ölçekler."""
        if self.current_resolution != (frame_w, frame_h):
            self._build_zones((frame_w, frame_h))

    def update(
        self,
        track_id: int,
        center: Tuple[int, int],
        frame_no: int,
        fps: float
    ) -> VehicleSpeedInfo:
        """
        Aracın anlık konumuna göre bölgelerle etkileşimini kontrol eder ve hız hesaplar.
        """
        if track_id not in self.results:
            self.results[track_id] = VehicleSpeedInfo(track_id=track_id)
        if track_id not in self.track_data:
            self.track_data[track_id] = {}

        info = self.results[track_id]
        if info.is_calculated:
            return info

        cx, cy = center
        effective_fps = fps if fps and fps > 0 else 30.0

        # ----------------------------------------------------------------------
        # 1. Mor Çizgi Koridor Modu (Plaka okuma bölgesiyle tam entegre hız hesabı)
        # ----------------------------------------------------------------------
        if self.use_corridor_mode:
            target_w, target_h = self.current_resolution if self.current_resolution else (1280, 720)
            y_top = int(target_h * self.band_top_ratio)
            y_bottom = int(target_h * self.band_bottom_ratio)
            band_h = max(20, abs(y_bottom - y_top))

            in_corridor = (y_top <= cy <= y_bottom)
            t_data = self.track_data[track_id]

            if in_corridor and "entry_frame" not in t_data:
                t_data["entry_frame"] = frame_no
                t_data["entry_pos"] = (cx, cy)
                t_data["entry_y"] = cy

            if "entry_frame" in t_data:
                entry_f = t_data["entry_frame"]
                entry_y = t_data["entry_y"]
                delta_y = abs(cy - entry_y)
                delta_f = max(1, frame_no - entry_f)
                elapsed = delta_f / effective_fps

                # Araç koridorda yeterli mesafe katettiğinde (min %20 veya 40 px) veya sınıra vardığında
                should_calc = (
                    (delta_y >= band_h * 0.20 and elapsed >= 0.20) or
                    (cy <= y_top + 15 or cy >= y_bottom - 15)
                )

                if should_calc and delta_y >= 30 and elapsed > 0:
                    dist_m = (delta_y / band_h) * self.corridor_real_distance_m
                    calc_speed = (dist_m / elapsed) * 3.6
                    speed_int = int(round(calc_speed))

                    if self.min_valid_speed <= speed_int <= self.max_valid_speed:
                        info.speed_kmh = speed_int
                        info.is_calculated = True
                        info.lane_name = "Sol Şerit" if cx < (target_w / 2) else "Sağ Şerit"
                        info.elapsed_time = round(elapsed, 2)
                        info.entry_frame = entry_f
                        info.exit_frame = frame_no
                        info.is_speeding = speed_int > self.speed_limit
                        t_data["speed"] = speed_int
                        return info

            return info

        # ----------------------------------------------------------------------
        # 2. Poligon Modu (Birim testler ve klasik kalibrasyonlar için)
        # ----------------------------------------------------------------------
        for zone in self.zones:
            z_data = self.track_data[track_id].setdefault(zone.name, {})

            # 1. Giriş bölgesine girdi mi?
            in_entry = cv2.pointPolygonTest(zone.entry_polygon, (float(cx), float(cy)), False) >= 0
            if in_entry and "entry_frame" not in z_data:
                z_data["entry_frame"] = frame_no
                z_data["entry_pos"] = (cx, cy)

            # 2. Giriş yapılmışsa ve çıkış bölgesine girdi mi?
            if "entry_frame" in z_data and "exit_frame" not in z_data:
                in_exit = cv2.pointPolygonTest(zone.exit_polygon, (float(cx), float(cy)), False) >= 0
                if in_exit:
                    entry_f = z_data["entry_frame"]
                    frame_diff = max(1, frame_no - entry_f)
                    elapsed = frame_diff / effective_fps
                    
                    entry_cx, entry_cy = z_data["entry_pos"]
                    
                    if zone.direction == "upward":
                        pixel_diff = float(entry_cy - cy)
                    elif zone.direction == "downward":
                        pixel_diff = float(cy - entry_cy)
                    else:
                        pixel_diff = float(np.hypot(cx - entry_cx, cy - entry_cy))

                    if pixel_diff > 0 and elapsed > 0:
                        # Mesafe oran-orantısı
                        speed_base = (zone.real_distance_m / elapsed) * 3.6  # m/s -> km/h
                        # Piksel farkı oran düzeltmesi
                        calculated_speed = speed_base * (pixel_diff / max(1.0, zone.calib_pixel_length))
                        speed_int = int(round(calculated_speed))

                        # Uç değerleri filtreleme
                        if self.min_valid_speed <= speed_int <= self.max_valid_speed:
                            info.speed_kmh = speed_int
                            info.is_calculated = True
                            info.lane_name = zone.name
                            info.elapsed_time = round(elapsed, 3)
                            info.entry_frame = entry_f
                            info.exit_frame = frame_no
                            info.is_speeding = speed_int > self.speed_limit
                            
                            z_data["exit_frame"] = frame_no
                            z_data["speed"] = speed_int
                            break

        return info

    def get_zones(self) -> List[SpeedZoneConfig]:
        return self.zones
