import sqlite3
import os
from typing import List, Dict, Optional, Any
from datetime import datetime


class TrafficDatabase:
    """
    Kalıcı SQLite Veritabanı Yönetim Katmanı (Persistent Storage Layer).
    Tüm tespit seanslarını, araç geçişlerini ve hız ihlallerini ilişkisel tabloda saklar.
    """
    def __init__(self, db_path: str = "output/traffic_surveillance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Tabloları ve indeksleri otomatik oluşturur."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_source TEXT NOT NULL,
                started_at TEXT NOT NULL,
                speed_limit INTEGER NOT NULL,
                total_vehicles INTEGER DEFAULT 0,
                total_violations INTEGER DEFAULT 0
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                speed_kmh INTEGER,
                speed_limit INTEGER NOT NULL,
                is_speeding BOOLEAN NOT NULL,
                plate_text TEXT,
                plate_confidence REAL,
                lane_name TEXT,
                elapsed_time_sec REAL,
                first_seen_frame INTEGER,
                last_seen_frame INTEGER,
                timestamp TEXT NOT NULL,
                snapshot_path TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate ON vehicle_records(plate_text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_speeding ON vehicle_records(is_speeding)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track ON vehicle_records(track_id)")
            conn.commit()
        finally:
            conn.close()

    def create_session(self, video_source: str, speed_limit: int) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO sessions (video_source, started_at, speed_limit)
            VALUES (?, ?, ?)
            """, (video_source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speed_limit))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def save_or_update_vehicle(self, record_dict: Dict[str, Any], session_id: Optional[int] = None):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            track_id = record_dict["track_id"]
            
            if session_id is not None:
                cursor.execute(
                    "SELECT id FROM vehicle_records WHERE session_id = ? AND track_id = ?",
                    (session_id, track_id)
                )
            else:
                cursor.execute(
                    "SELECT id FROM vehicle_records WHERE track_id = ? ORDER BY id DESC LIMIT 1",
                    (track_id,)
                )
            row = cursor.fetchone()

            if row:
                cursor.execute("""
                UPDATE vehicle_records SET
                    class_name = ?,
                    speed_kmh = ?,
                    is_speeding = ?,
                    plate_text = ?,
                    plate_confidence = ?,
                    lane_name = ?,
                    elapsed_time_sec = ?,
                    last_seen_frame = ?,
                    snapshot_path = COALESCE(?, snapshot_path)
                WHERE id = ?
                """, (
                    record_dict.get("class_name", "vehicle"),
                    record_dict.get("speed_kmh"),
                    1 if record_dict.get("is_speeding") else 0,
                    record_dict.get("plate_text"),
                    record_dict.get("plate_confidence"),
                    record_dict.get("lane_name"),
                    record_dict.get("elapsed_time_sec"),
                    record_dict.get("last_seen_frame"),
                    record_dict.get("snapshot_path"),
                    row["id"]
                ))
            else:
                cursor.execute("""
                INSERT INTO vehicle_records (
                    session_id, track_id, class_name, speed_kmh, speed_limit,
                    is_speeding, plate_text, plate_confidence, lane_name,
                    elapsed_time_sec, first_seen_frame, last_seen_frame,
                    timestamp, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    track_id,
                    record_dict.get("class_name", "vehicle"),
                    record_dict.get("speed_kmh"),
                    record_dict.get("speed_limit", 80),
                    1 if record_dict.get("is_speeding") else 0,
                    record_dict.get("plate_text"),
                    record_dict.get("plate_confidence"),
                    record_dict.get("lane_name"),
                    record_dict.get("elapsed_time_sec"),
                    record_dict.get("first_seen_frame", 1),
                    record_dict.get("last_seen_frame", 1),
                    record_dict.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    record_dict.get("snapshot_path")
                ))
            conn.commit()
        finally:
            conn.close()

    def get_all_records(self, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("SELECT * FROM vehicle_records WHERE session_id = ? ORDER BY id DESC", (session_id,))
            else:
                cursor.execute("SELECT * FROM vehicle_records ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def get_violations(self, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT * FROM vehicle_records WHERE session_id = ? AND is_speeding = 1 ORDER BY id DESC",
                    (session_id,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM vehicle_records WHERE is_speeding = 1 ORDER BY id DESC"
                )
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def search_by_plate(self, plate_query: str) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM vehicle_records WHERE plate_text LIKE ? ORDER BY id DESC",
                (f"%{plate_query}%",)
            )
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def update_plate_text(self, record_id: int, new_plate: str) -> bool:
        """Kullanıcının yanlış okunan plakayı arayüz üzerinden elle düzeltmesini sağlar."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE vehicle_records SET plate_text = ? WHERE id = ?",
                (new_plate.strip().upper(), record_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
