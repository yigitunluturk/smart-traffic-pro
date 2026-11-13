import sqlite3
import os
from typing import List, Dict, Optional, Any
from datetime import datetime


class TrafficDatabase:

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

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL UNIQUE,
                reason TEXT NOT NULL,
                severity TEXT DEFAULT 'danger',
                added_at TEXT NOT NULL
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS plate_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                old_plate TEXT,
                new_plate TEXT NOT NULL,
                corrected_at TEXT NOT NULL,
                FOREIGN KEY (record_id) REFERENCES vehicle_records(id)
            )
            """)
            cursor.execute("PRAGMA table_info(vehicle_records)")
            cols = {r[1] for r in cursor.fetchall()}
            for col, typ in (("plate_candidate", "TEXT"), ("plate_status", "TEXT")):
                if col not in cols:
                    cursor.execute(f"ALTER TABLE vehicle_records ADD COLUMN {col} {typ}")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate ON vehicle_records(plate_text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_speeding ON vehicle_records(is_speeding)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_track ON vehicle_records(track_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_plate ON watchlist(plate_text)")
            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM watchlist")
            if cursor.fetchone()[0] == 0:
                demo_watchlist = [
                    ("34VF5821", "Çalıntı İhbarı / Asayiş Şube", "danger", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    ("34BMS11", "Hacizli Yakalamalı Araç / İcra Dairesi", "warning", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    ("34ABC01", "Trafikten Men / Sahte Plaka Şüphesi", "danger", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                ]
                cursor.executemany(
                    "INSERT OR IGNORE INTO watchlist (plate_text, reason, severity, added_at) VALUES (?, ?, ?, ?)",
                    demo_watchlist
                )
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
                    snapshot_path = COALESCE(?, snapshot_path),
                    plate_candidate = ?,
                    plate_status = CASE WHEN plate_status = 'corrected' THEN plate_status ELSE ? END
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
                    record_dict.get("plate_candidate"),
                    record_dict.get("plate_status"),
                    row["id"]
                ))
            else:
                cursor.execute("""
                INSERT INTO vehicle_records (
                    session_id, track_id, class_name, speed_kmh, speed_limit,
                    is_speeding, plate_text, plate_confidence, lane_name,
                    elapsed_time_sec, first_seen_frame, last_seen_frame,
                    timestamp, snapshot_path, plate_candidate, plate_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record_dict.get("snapshot_path"),
                    record_dict.get("plate_candidate"),
                    record_dict.get("plate_status")
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

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vehicle_records WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_record_by_snapshot(self, filename: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vehicle_records WHERE snapshot_path LIKE ? ORDER BY id DESC LIMIT 1", (f"%{filename}%",))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_watchlist(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watchlist ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def add_to_watchlist(self, plate_text: str, reason: str, severity: str = "danger") -> bool:
        norm_plate = plate_text.strip().upper().replace(" ", "")
        if not norm_plate:
            return False
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO watchlist (plate_text, reason, severity, added_at)
            VALUES (?, ?, ?, ?)
            """, (norm_plate, reason.strip(), severity, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            return True
        except Exception as e:
            print(f"[DB] Kara liste ekleme hatası: {e}")
            return False
        finally:
            conn.close()

    def remove_from_watchlist(self, item_id: int) -> bool:
        """Kara listeden kayıt siler."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watchlist WHERE id = ?", (item_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def check_watchlist(self, plate_text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not plate_text:
            return None
        norm_plate = plate_text.strip().upper().replace(" ", "")
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watchlist")
            for row in cursor.fetchall():
                w_plate = row["plate_text"].strip().upper().replace(" ", "")
                if w_plate == norm_plate or (len(w_plate) >= 5 and w_plate in norm_plate):
                    return dict(row)
            return None
        finally:
            conn.close()

    def get_analytics_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT speed_kmh, class_name, is_speeding FROM vehicle_records WHERE speed_kmh IS NOT NULL")
            rows = cursor.fetchall()

            speed_dist = {
                "0-50 km/s": 0,
                "51-70 km/s": 0,
                "71-80 km/s": 0,
                "81-100 km/s": 0,
                "100+ km/s": 0
            }
            class_dist = {
                "Otomobil": 0,
                "Kamyon": 0,
                "Otobüs": 0,
                "Motosiklet": 0
            }
            class_map = {
                "car": "Otomobil",
                "truck": "Kamyon",
                "bus": "Otobüs",
                "motorcycle": "Motosiklet"
            }

            speeds = []
            viol_count = 0

            for r in rows:
                spd = r["speed_kmh"]
                if spd is not None and spd > 0:
                    speeds.append(spd)
                    if spd <= 50:
                        speed_dist["0-50 km/s"] += 1
                    elif spd <= 70:
                        speed_dist["51-70 km/s"] += 1
                    elif spd <= 80:
                        speed_dist["71-80 km/s"] += 1
                    elif spd <= 100:
                        speed_dist["81-100 km/s"] += 1
                    else:
                        speed_dist["100+ km/s"] += 1

                c_name = class_map.get((r["class_name"] or "").lower(), "Otomobil")
                class_dist[c_name] = class_dist.get(c_name, 0) + 1

                if r["is_speeding"]:
                    viol_count += 1

            total = len(rows)
            avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
            max_speed = max(speeds) if speeds else 0
            viol_rate = round((viol_count / total * 100), 1) if total > 0 else 0.0

            return {
                "total_vehicles": total,
                "total_violations": viol_count,
                "violation_rate": viol_rate,
                "average_speed": avg_speed,
                "max_speed": max_speed,
                "speed_distribution": speed_dist,
                "class_distribution": class_dist
            }
        finally:
            conn.close()

    def clear_records(self) -> bool:
        """Tüm araç ve ihlal kayıtlarını veritabanından temizler."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vehicle_records")
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            return True
        except Exception as e:
            print(f"[DB] Temizleme hatası: {e}")
            return False
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # FR-08: Arama / filtreleme
    # ------------------------------------------------------------------
    def query_records(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        plate_query: Optional[str] = None,
        lane: Optional[str] = None,
        violations_only: bool = False,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Tarih aralığı (YYYY-MM-DD), plaka alt dizesi ve şeride göre kayıt arar."""
        sql = "SELECT * FROM vehicle_records WHERE 1=1"
        args: List[Any] = []
        if date_from:
            sql += " AND timestamp >= ?"
            args.append(f"{date_from} 00:00:00")
        if date_to:
            sql += " AND timestamp <= ?"
            args.append(f"{date_to} 23:59:59")
        if plate_query:
            q = plate_query.strip().upper().replace(" ", "")
            sql += " AND (plate_text LIKE ? OR plate_candidate LIKE ?)"
            args += [f"%{q}%", f"%{q}%"]
        if lane:
            sql += " AND lane_name = ?"
            args.append(lane)
        if violations_only:
            sql += " AND is_speeding = 1"
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_lanes(self) -> List[str]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT lane_name FROM vehicle_records WHERE lane_name IS NOT NULL ORDER BY lane_name")
            return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # FR-09: Doğrulamalı plaka düzeltme + denetim kaydı
    # ------------------------------------------------------------------
    def correct_plate(self, record_id: int, new_plate: str, validator=None):

        norm = "".join(ch for ch in (new_plate or "").upper() if ch.isalnum())
        if not norm:
            return False, "Plaka boş olamaz."
        if validator is not None and not validator(norm):
            return False, f"'{norm}' Türk plaka formatına uymuyor (il kodu 01-81, 1-3 harf, 2-4 rakam; Q/W/X yok)."
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT plate_text, plate_candidate FROM vehicle_records WHERE id = ?", (record_id,))
            row = cur.fetchone()
            if row is None:
                return False, "Kayıt bulunamadı."
            old = row["plate_text"] or row["plate_candidate"]
            cur.execute(
                "UPDATE vehicle_records SET plate_text = ?, plate_status = 'corrected' WHERE id = ?",
                (norm, record_id),
            )
            cur.execute(
                "INSERT INTO plate_corrections (record_id, old_plate, new_plate, corrected_at) VALUES (?, ?, ?, ?)",
                (record_id, old, norm, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            return True, f"Plaka güncellendi: {old or '-'} → {norm}"
        finally:
            conn.close()

    def get_corrections(self, record_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if record_id is None:
                cur.execute("SELECT * FROM plate_corrections ORDER BY id DESC")
            else:
                cur.execute("SELECT * FROM plate_corrections WHERE record_id = ? ORDER BY id DESC", (record_id,))
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # FR-07 / NFR-5: Saklama politikası (KVKK)
    # ------------------------------------------------------------------
    def apply_retention(self, days: int = 30, now: Optional[datetime] = None) -> int:
        """
        İhlal yapmayan geçişlerin plaka metnini, saklama süresi dolunca siler.
        İhlal kayıtları (kanıt) korunur. Silinen kayıt sayısını döner.
        """
        from datetime import timedelta
        now = now or datetime.now()
        cutoff = (now - timedelta(days=int(days))).strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """UPDATE vehicle_records
                   SET plate_text = NULL, plate_candidate = NULL, plate_confidence = NULL,
                       plate_status = 'purged'
                   WHERE is_speeding = 0 AND timestamp < ?
                     AND (plate_text IS NOT NULL OR plate_candidate IS NOT NULL)""",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

