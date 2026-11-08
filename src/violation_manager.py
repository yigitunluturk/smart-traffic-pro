from dataclasses import dataclass, asdict
from datetime import datetime
import os
from typing import Dict, List, Optional
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

        if self.save_csv:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if self.save_snapshots:
            os.makedirs(self.violations_dir, exist_ok=True)

    def start_session(self, video_source: str):
        self.current_session_id = self.db.create_session(video_source, self.speed_limit)

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
        vehicle_box: Optional[tuple] = None
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

        if rec.is_speeding and track_id not in self.saved_snapshots and self.save_snapshots:
            if frame_for_snapshot is not None:
                snap_path = self._save_violation_snapshot(rec, frame_for_snapshot, vehicle_box)
                rec.snapshot_path = snap_path
                self.saved_snapshots.add(track_id)

        self.db.save_or_update_vehicle(asdict(rec), session_id=self.current_session_id)

        return rec

    def _save_violation_snapshot(
        self,
        rec: VehicleRecord,
        frame: np.ndarray,
        vehicle_box: Optional[tuple]
    ) -> str:
        plate_str = rec.plate_text or "UNKNOWN"
        filename = f"violation_id{rec.track_id}_{plate_str}_{rec.speed_kmh}kmh.jpg"
        filepath = os.path.join(self.violations_dir, filename)

        snap = frame.copy()
        if vehicle_box is not None:
            x1, y1, x2, y2 = vehicle_box
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            header = f"SPEED VIOLATION: {rec.speed_kmh} km/h (Limit: {rec.speed_limit}) | Plate: {plate_str}"
            cv2.putText(snap, header, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite(filepath, snap)
        return filepath

    def export_csv(self) -> str:
        """Kayıtları CSV dosyasına aktarır."""
        if not self.records:
            return self.csv_path
        
        df = pd.DataFrame([asdict(r) for r in self.records.values()])
        df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
        return self.csv_path

    def get_summary(self) -> Dict:
        """İstatistik özetini döner."""
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
