from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    provisional_kmh: Optional[int] = None 
    distance_m: Optional[float] = None   
    n_samples: int = 0


@dataclass
class _TrackSamples:
    t: List[float] = field(default_factory=list)
    X: List[float] = field(default_factory=list)
    Y: List[float] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    last_frame: int = -1


def theil_sen_slope(t: np.ndarray, v: np.ndarray, max_pairs: int = 4000) -> float:
    """Medyan ikili eğim (Theil–Sen). Aykırı değerlere %29'a kadar dayanıklıdır."""
    n = len(t)
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        sel = np.linspace(0, len(i) - 1, max_pairs).astype(int)
        i, j = i[sel], j[sel]
    dt = t[j] - t[i]
    ok = dt > 1e-6
    if not np.any(ok):
        return 0.0
    return float(np.median((v[j][ok] - v[i][ok]) / dt[ok]))


class SpeedEstimator:

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
        use_corridor_mode: bool = False,
        homography_config: Optional[Dict[str, Any]] = None,
        min_track_time: float = 0.6,
        min_track_distance_m: float = 5.0,
        lock_time: float = 1.2,
        lock_distance_m: float = 25.0,
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

        self.min_track_time = min_track_time
        self.min_track_distance_m = min_track_distance_m
        self.lock_time = lock_time
        self.lock_distance_m = lock_distance_m
        self.H: Optional[np.ndarray] = None            
        self.H_inv: Optional[np.ndarray] = None
        self.homography_cfg: Dict[str, Any] = {}
        self.calib_base_res: Tuple[int, int] = tuple(base_resolution or (1280, 720))
        self.measure_y_range: Tuple[float, float] = (0.0, 1.0)
        self.lanes: List[Dict[str, Any]] = []
        self.samples: Dict[int, _TrackSamples] = {}
        if homography_config:
            self.set_homography(homography_config)

        if self.raw_zones_config:
            self._build_zones(self.base_resolution or (1280, 720))

        # Araç durum takibi: {track_id: {zone_name: {'entry_frame', 'entry_pos', ...}}}
        self.track_data: Dict[int, Dict[str, Any]] = {}
        # Nihai hesaplanan hız sonuçları: {track_id: VehicleSpeedInfo}
        self.results: Dict[int, VehicleSpeedInfo] = {}

    # ------------------------------------------------------------------
    # Kalibrasyon
    # ------------------------------------------------------------------
    @property
    def method(self) -> str:
        if self.H is not None:
            return "homography"
        return "corridor" if self.use_corridor_mode else "zones"

    def set_homography(self, cfg: Dict[str, Any]):
        img = np.array(cfg["image_points"], dtype=np.float64).reshape(-1, 2)
        wld = np.array(cfg["world_points"], dtype=np.float64).reshape(-1, 2)
        if len(img) < 4 or len(img) != len(wld):
            raise ValueError("Homografi için en az 4 eşleşen nokta gerekir.")
        H, _ = cv2.findHomography(img, wld, 0 if len(img) == 4 else cv2.RANSAC)
        if H is None:
            raise ValueError("Homografi hesaplanamadı (noktalar doğrusal olabilir).")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.homography_cfg = dict(cfg)
        self.calib_base_res = tuple(cfg.get("base_resolution", self.base_resolution or (1280, 720)))
        yr = cfg.get("measure_y_range", [0.0, 1.0])
        self.measure_y_range = (float(yr[0]), float(yr[1]))
        self.lanes = list(cfg.get("lanes", []))

    def image_to_world(self, pts_native: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        """Referans karedeki (tam çözünürlük) noktaları metre cinsinden yol koordinatına çevirir."""
        w, h = frame_size
        bw, bh = self.calib_base_res
        p = np.asarray(pts_native, dtype=np.float64).reshape(-1, 2) * [bw / w, bh / h]
        ph = np.hstack([p, np.ones((len(p), 1))]) @ self.H.T
        return ph[:, :2] / ph[:, 2:3]

    def world_to_image(self, pts_world: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        w, h = frame_size
        bw, bh = self.calib_base_res
        p = np.asarray(pts_world, dtype=np.float64).reshape(-1, 2)
        ph = np.hstack([p, np.ones((len(p), 1))]) @ self.H_inv.T
        return ph[:, :2] / ph[:, 2:3] * [w / bw, h / bh]

    def _lane_for(self, X: float, cx: int, frame_w: int) -> str:
        for ln in self.lanes:
            if float(ln["x_min"]) <= X < float(ln["x_max"]):
                return ln["name"]
        return "Sol Şerit" if cx < frame_w / 2 else "Sağ Şerit"

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

            scaled_entry = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in raw_entry]
            scaled_exit = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in raw_exit]

            entry_poly = np.array(scaled_entry, dtype=np.int32).reshape((-1, 1, 2))
            exit_poly = np.array(scaled_exit, dtype=np.int32).reshape((-1, 1, 2))

            calib_px = float(z.get("calib_pixel_length", 80.0)) * scale_y

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
                    color_entry=tuple(z.get("color_entry", [0, 255, 255])),
                    color_exit=tuple(z.get("color_exit", [0, 0, 255])),
                )
            )

    def adapt_to_frame(self, frame_w: int, frame_h: int):
        if self.current_resolution != (frame_w, frame_h):
            if self.raw_zones_config:
                self._build_zones((frame_w, frame_h))
            else:
                self.current_resolution = (frame_w, frame_h)

    # ------------------------------------------------------------------
    # Ana güncelleme
    # ------------------------------------------------------------------
    def update(
        self,
        track_id: int,
        center: Tuple[int, int],
        frame_no: int,
        fps: float,
        ground_point_ref: Optional[Tuple[float, float]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> VehicleSpeedInfo:

        if track_id not in self.results:
            self.results[track_id] = VehicleSpeedInfo(track_id=track_id)
        if track_id not in self.track_data:
            self.track_data[track_id] = {}

        info = self.results[track_id]
        if info.is_calculated:
            return info

        effective_fps = fps if fps and fps > 0 else 30.0

        if self.H is not None:
            return self._update_homography(info, track_id, center, frame_no, effective_fps,
                                           ground_point_ref, box)
        if self.use_corridor_mode:
            return self._update_corridor(info, track_id, center, frame_no, effective_fps)
        return self._update_zones(info, track_id, center, frame_no, effective_fps)

    # ---------------------------- Homografi ---------------------------
    def _update_homography(self, info, track_id, center, frame_no, fps, ground_pt, box):
        fw, fh = self.current_resolution or self.calib_base_res
        if ground_pt is None:
            if box is not None:
                ground_pt = ((box[0] + box[2]) / 2.0, float(box[3]))
            else:
                ground_pt = (float(center[0]), float(center[1]))

        if box is not None:
            x1, y1, x2, y2 = box
            margin = max(2, int(0.004 * fh))
            if y2 >= fh - margin or x1 <= margin or x2 >= fw - margin:
                return info

        gy_ratio = ground_pt[1] / float(fh)
        if not (self.measure_y_range[0] <= gy_ratio <= self.measure_y_range[1]):
            return info

        X, Y = self.image_to_world(np.array([ground_pt]), (fw, fh))[0]
        if self.lanes and not any(float(ln["x_min"]) <= X < float(ln["x_max"]) for ln in self.lanes):
            return info
        s = self.samples.setdefault(track_id, _TrackSamples())
        if s.frames and frame_no - s.last_frame > int(fps * 1.0):
            self.samples[track_id] = s = _TrackSamples()
        s.t.append(frame_no / fps)
        s.X.append(float(X))
        s.Y.append(float(Y))
        s.frames.append(frame_no)
        s.last_frame = frame_no
        if not info.lane_name:
            info.lane_name = self._lane_for(float(X), int(center[0]), fw)

        est = self._estimate(s)
        if est is None:
            return info
        speed, span_t, dist = est
        info.provisional_kmh = int(round(speed))
        info.n_samples = len(s.t)
        if span_t >= self.lock_time or dist >= self.lock_distance_m:
            self._lock(info, s, speed, span_t, dist)
        return info

    def _estimate(self, s: _TrackSamples, min_time=None, min_dist=None):
        min_time = self.min_track_time if min_time is None else min_time
        min_dist = self.min_track_distance_m if min_dist is None else min_dist
        if len(s.t) < 4:
            return None
        t = np.array(s.t)
        X = np.array(s.X)
        Y = np.array(s.Y)
        span_t = t[-1] - t[0]
        if span_t < min_time:
            return None
        vx = theil_sen_slope(t, X)
        vy = theil_sen_slope(t, Y)
        v = float(np.hypot(vx, vy))
        dist = v * span_t
        if dist < min_dist:
            return None
        kmh = v * 3.6
        if not (self.min_valid_speed <= kmh <= self.max_valid_speed):
            return None
        return kmh, span_t, dist

    def _lock(self, info: VehicleSpeedInfo, s: _TrackSamples, speed: float, span_t: float, dist: float):
        info.speed_kmh = int(round(speed))
        info.provisional_kmh = info.speed_kmh
        info.is_calculated = True
        info.elapsed_time = round(span_t, 2)
        info.distance_m = round(dist, 1)
        info.entry_frame = s.frames[0]
        info.exit_frame = s.frames[-1]
        info.n_samples = len(s.t)
        info.is_speeding = info.speed_kmh > self.speed_limit
        self.track_data.setdefault(info.track_id, {})["speed"] = info.speed_kmh

    def finalize_track(self, track_id: int) -> Optional[VehicleSpeedInfo]:

        info = self.results.get(track_id)
        s = self.samples.pop(track_id, None)
        if info is None or info.is_calculated or s is None:
            return info
        est = self._estimate(s, min_time=max(0.4, self.min_track_time * 0.7),
                             min_dist=max(3.0, self.min_track_distance_m * 0.7))
        if est is not None:
            self._lock(info, s, *est)
        return info

    # ----------------------------- Koridor ----------------------------
    def _update_corridor(self, info, track_id, center, frame_no, fps):
        cx, cy = center
        target_w, target_h = self.current_resolution if self.current_resolution else (1280, 720)
        y_top = int(target_h * self.band_top_ratio)
        y_bottom = int(target_h * self.band_bottom_ratio)
        band_h = max(20, abs(y_bottom - y_top))

        in_corridor = (y_top <= cy <= y_bottom)
        t_data = self.track_data[track_id]

        if in_corridor and "entry_frame" not in t_data:
            t_data["entry_frame"] = frame_no
            t_data["entry_y"] = cy

        if "entry_frame" in t_data:
            entry_f = t_data["entry_frame"]
            delta_y = abs(cy - t_data["entry_y"])
            elapsed = max(1, frame_no - entry_f) / fps
            should_calc = (
                (delta_y >= band_h * 0.20 and elapsed >= 0.20) or
                (cy <= y_top + 15 or cy >= y_bottom - 15)
            )
            if should_calc and delta_y >= 30 and elapsed > 0:
                dist_m = (delta_y / band_h) * self.corridor_real_distance_m
                speed_int = int(round((dist_m / elapsed) * 3.6))
                if self.min_valid_speed <= speed_int <= self.max_valid_speed:
                    info.speed_kmh = speed_int
                    info.provisional_kmh = speed_int
                    info.is_calculated = True
                    info.lane_name = "Sol Şerit" if cx < (target_w / 2) else "Sağ Şerit"
                    info.elapsed_time = round(elapsed, 2)
                    info.entry_frame = entry_f
                    info.exit_frame = frame_no
                    info.is_speeding = speed_int > self.speed_limit
                    t_data["speed"] = speed_int
        return info

    # ------------------------------ Bölge -----------------------------
    def _update_zones(self, info, track_id, center, frame_no, fps):
        cx, cy = center
        for zone in self.zones:
            z_data = self.track_data[track_id].setdefault(zone.name, {})

            in_entry = cv2.pointPolygonTest(zone.entry_polygon, (float(cx), float(cy)), False) >= 0
            if in_entry and "entry_frame" not in z_data:
                z_data["entry_frame"] = frame_no
                z_data["entry_pos"] = (cx, cy)

            if "entry_frame" in z_data and "exit_frame" not in z_data:
                in_exit = cv2.pointPolygonTest(zone.exit_polygon, (float(cx), float(cy)), False) >= 0
                if in_exit:
                    entry_f = z_data["entry_frame"]
                    elapsed = max(1, frame_no - entry_f) / fps
                    entry_cx, entry_cy = z_data["entry_pos"]

                    if zone.direction == "upward":
                        pixel_diff = float(entry_cy - cy)
                    elif zone.direction == "downward":
                        pixel_diff = float(cy - entry_cy)
                    else:
                        pixel_diff = float(np.hypot(cx - entry_cx, cy - entry_cy))

                    if pixel_diff > 0 and elapsed > 0:
                        speed_base = (zone.real_distance_m / elapsed) * 3.6
                        calculated_speed = speed_base * (pixel_diff / max(1.0, zone.calib_pixel_length))
                        speed_int = int(round(calculated_speed))
                        if self.min_valid_speed <= speed_int <= self.max_valid_speed:
                            info.speed_kmh = speed_int
                            info.provisional_kmh = speed_int
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
