from dataclasses import dataclass, asdict
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from .database import TrafficDatabase


@dataclass
class VehicleRecord:
    track_id: int
    class_name: str
    speed_kmh: Optional[int]
    speed_limit: int
    is_speeding: bool
    plate_text: Optional[str]
    plate_confidence: Optional[float]
    lane_name: Optional[str]
    elapsed_time_sec: Optional[float]
    first_seen_frame: int
    last_seen_frame: int
    timestamp: str
    snapshot_path: Optional[str] = None
    plate_candidate: Optional[str] = None   
    plate_status: Optional[str] = None     


class ViolationManager:

    def __init__(
        self,
        speed_limit: int = 80,
        save_csv: bool = True,
        csv_path: str = "output/arac_kayitlari.csv",
        save_snapshots: bool = True,
        violations_dir: str = "output/ihlaller",
        db_path: str = "output/traffic_surveillance.db"
    ):
        self.speed_limit = speed_limit
        self.save_csv = save_csv
        self.csv_path = csv_path
        self.save_snapshots = save_snapshots
        self.violations_dir = violations_dir
        self.db = TrafficDatabase(db_path=db_path)
        self.current_session_id = None

        self.records: Dict[int, VehicleRecord] = {}
        self.saved_snapshots: set = set()
        self.pending_evidence: Dict[int, Tuple[np.ndarray, tuple, int]] = {}
        self._db_signature: Dict[int, tuple] = {}
        self.plate_rel: Dict[int, tuple] = {}

        if self.save_csv:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if self.save_snapshots:
            os.makedirs(self.violations_dir, exist_ok=True)

    def start_session(self, video_source: str):
        self.current_session_id = self.db.create_session(video_source, self.speed_limit)

    # ------------------------------------------------------------------
    def register_or_update(
        self,
        track_id: int,
        class_name: str,
        frame_no: int,
        speed_kmh: Optional[int] = None,
        lane_name: Optional[str] = None,
        elapsed_time: Optional[float] = None,
        plate_text: Optional[str] = None,
        plate_conf: Optional[float] = None,
        frame_for_snapshot: Optional[np.ndarray] = None,
        vehicle_box: Optional[tuple] = None,
        plate_final: bool = True,
        plate_candidate: Optional[str] = None,
        plate_rel_box: Optional[tuple] = None,
    ) -> VehicleRecord:

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if track_id not in self.records:
            is_speeding = (speed_kmh > self.speed_limit) if speed_kmh is not None else False
            self.records[track_id] = VehicleRecord(
                track_id=track_id,
                class_name=class_name,
                speed_kmh=speed_kmh,
                speed_limit=self.speed_limit,
                is_speeding=is_speeding,
                plate_text=plate_text,
                plate_confidence=plate_conf,
                lane_name=lane_name,
                elapsed_time_sec=elapsed_time,
                first_seen_frame=frame_no,
                last_seen_frame=frame_no,
                timestamp=now_str
            )
        else:
            rec = self.records[track_id]
            rec.last_seen_frame = frame_no
            if class_name:
                rec.class_name = class_name
            if speed_kmh is not None:
                rec.speed_kmh = speed_kmh
                rec.is_speeding = speed_kmh > self.speed_limit
            if lane_name:
                rec.lane_name = lane_name
            if elapsed_time is not None:
                rec.elapsed_time_sec = elapsed_time
            if plate_text:
                rec.plate_text = plate_text
                rec.plate_confidence = plate_conf

        rec = self.records[track_id]
        if plate_rel_box is not None:
            self.plate_rel[track_id] = plate_rel_box
        if plate_candidate:
            rec.plate_candidate = plate_candidate
        if rec.plate_text:
            rec.plate_status = "valid"
        elif rec.plate_candidate:
            rec.plate_status = "low_confidence"

        if rec.is_speeding and track_id not in self.saved_snapshots and self.save_snapshots:
            if frame_for_snapshot is not None:
                if rec.plate_text and plate_final:
                    rec.snapshot_path = self._save_violation_snapshot(rec, frame_for_snapshot, vehicle_box)
                    self.saved_snapshots.add(track_id)
                    self.pending_evidence.pop(track_id, None)
                else:
                    area = 0
                    if vehicle_box is not None:
                        area = max(0, vehicle_box[2] - vehicle_box[0]) * max(0, vehicle_box[3] - vehicle_box[1])
                    prev = self.pending_evidence.get(track_id)
                    if prev is None or area > prev[2]:
                        self.pending_evidence[track_id] = (frame_for_snapshot.copy(), vehicle_box, area)

        self._persist(rec)
        return rec

    def finalize_track(self, track_id: int, plate_text: Optional[str] = None,
                       plate_conf: Optional[float] = None, speed_info=None):
        rec = self.records.get(track_id)
        if rec is None:
            return None
        if speed_info is not None and speed_info.speed_kmh is not None:
            rec.speed_kmh = speed_info.speed_kmh
            rec.is_speeding = speed_info.speed_kmh > self.speed_limit
            rec.elapsed_time_sec = speed_info.elapsed_time
            if speed_info.lane_name:
                rec.lane_name = speed_info.lane_name
        if plate_text:
            rec.plate_text = plate_text
            rec.plate_confidence = plate_conf
            rec.plate_status = "valid"
        pend = self.pending_evidence.pop(track_id, None)
        if rec.is_speeding and track_id not in self.saved_snapshots and self.save_snapshots and pend is not None:
            rec.snapshot_path = self._save_violation_snapshot(rec, pend[0], pend[1])
            self.saved_snapshots.add(track_id)
        self._persist(rec, force=True)
        return rec

    def flush(self):
        for tid in list(self.pending_evidence.keys()):
            self.finalize_track(tid)

    # ------------------------------------------------------------------
    def _persist(self, rec: VehicleRecord, force: bool = False):
        sig = (rec.class_name, rec.speed_kmh, rec.is_speeding, rec.plate_text,
               rec.plate_candidate, rec.lane_name, rec.snapshot_path)
        if not force and self._db_signature.get(rec.track_id) == sig:
            return
        self._db_signature[rec.track_id] = sig
        try:
            self.db.save_or_update_vehicle(asdict(rec), session_id=self.current_session_id)
        except Exception as e:
            print(f"[ViolationManager] Veritabanı yazma uyarısı: {e}")

    def _save_violation_snapshot(
        self,
        rec: VehicleRecord,
        frame: np.ndarray,
        vehicle_box: Optional[tuple]
    ) -> str:
        plate_str = rec.plate_text or "OKUNAMADI"
        filename = f"violation_id{rec.track_id}_{plate_str}_{rec.speed_kmh}kmh.jpg"
        filepath = os.path.join(self.violations_dir, filename)

        snap = frame.copy()
        if vehicle_box is not None:
            x1, y1, x2, y2 = vehicle_box
            h = snap.shape[0]
            fs = max(0.8, h / 1080.0 * 0.9)
            th = max(2, int(h / 540))
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), th + 1)
            header = f"HIZ IHLALI: {rec.speed_kmh} km/s (Sinir: {rec.speed_limit}) | Plaka: {plate_str}"
            cv2.putText(snap, header, (max(5, x1), max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th, cv2.LINE_AA)

        cv2.imwrite(filepath, snap)
        self._save_plate_crop(rec.track_id, frame, vehicle_box, filepath)
        return filepath

    def _save_plate_crop(self, track_id: int, frame: np.ndarray, vehicle_box: Optional[tuple], snap_path: str):
        rel = self.plate_rel.get(track_id)
        if rel is None or vehicle_box is None:
            return None
        x1, y1, x2, y2 = vehicle_box
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        px1, py1 = x1 + rel[0] * w, y1 + rel[1] * h
        px2, py2 = x1 + rel[2] * w, y1 + rel[3] * h
        mw, mh = 0.25 * (px2 - px1), 0.4 * (py2 - py1)
        H, W = frame.shape[:2]
        cx1, cy1 = int(max(0, px1 - mw)), int(max(0, py1 - mh))
        cx2, cy2 = int(min(W, px2 + mw)), int(min(H, py2 + mh))
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.shape[1] < 320:
            f = 320.0 / crop.shape[1]
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        out = snap_path[:-4] + "_plaka.jpg"
        cv2.imwrite(out, crop)
        return out

    def export_csv(self) -> str:
        if not self.records:
            return self.csv_path
        df = pd.DataFrame([asdict(r) for r in self.records.values()])
        df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
        return self.csv_path

    def get_summary(self) -> Dict:
        total = len(self.records)
        with_speed = [r for r in self.records.values() if r.speed_kmh is not None]
        with_plate = [r for r in self.records.values() if r.plate_text is not None]
        violations = [r for r in self.records.values() if r.is_speeding]

        speeds = [r.speed_kmh for r in with_speed if r.speed_kmh is not None]
        avg_speed = round(float(np.mean(speeds)), 1) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0

        return {
            "total_vehicles": total,
            "speed_measured_count": len(with_speed),
            "plate_recognized_count": len(with_plate),
            "violations_count": len(violations),
            "average_speed": avg_speed,
            "max_speed": max_speed
        }
