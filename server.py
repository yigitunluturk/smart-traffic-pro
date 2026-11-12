import copy
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import imageio_ffmpeg
import pandas as pd
import yaml
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.pipeline import TrafficPipeline
from src.database import TrafficDatabase
from src.report_generator import generate_violation_pdf

# Göreli yolların (config, output, web) her zaman proje klasörüne göre çözülmesi için
# LEGACY / UNSUPPORTED: Resmi arayüz Streamlit (app.py). Bu panel yalnızca arşiv amaçlıdır.
# Çalıştırma: proje kökünden  python legacy/server.py
import sys
LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(LEGACY_DIR)
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

app = Flask(__name__, static_folder=os.path.join(LEGACY_DIR, "web"), static_url_path="")
CORS(app)

# Global Durum
CONFIG_PATH = "config/default_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OUTPUT_CFG = config.get("output", {})
DB_PATH = OUTPUT_CFG.get("database_path", "output/traffic_surveillance.db")
VIOLATIONS_DIR = OUTPUT_CFG.get("violations_dir", "output/ihlaller")
UPLOAD_DIR = "output/uploads"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

db = TrafficDatabase(db_path=DB_PATH)


def _default_video_path() -> str:
    """Varsayılan video: demo videosu, yoksa yüklenenlerden ilki."""
    preferred = os.path.join(UPLOAD_DIR, "demo_part1_100frames.mp4")
    if os.path.exists(preferred):
        return preferred
    if os.path.isdir(UPLOAD_DIR):
        for f in sorted(os.listdir(UPLOAD_DIR)):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                return os.path.join(UPLOAD_DIR, f)
    return preferred


def _build_run_config(speed_limit: int) -> dict:
    run_cfg = copy.deepcopy(config)
    run_cfg.setdefault("speed", {})["speed_limit"] = int(speed_limit)
    return run_cfg


def _empty_stats() -> dict:
    return {
        "total_vehicles": 0,
        "speed_measured_count": 0,
        "plate_recognized_count": 0,
        "violations_count": 0,
        "average_speed": 0.0,
        "max_speed": 0,
        "fps": 0.0,
        "frame_no": 0,
        "is_running": False,
        "is_paused": False,
        "error": None,
        "watchlist_alerts": [],
        "latest_alert": None
    }


class StreamController:
    def __init__(self):
        self.pipeline: Optional[TrafficPipeline] = None
        self.is_running = False
        self.is_paused = False
        self.current_video_path = _default_video_path()
        self.speed_limit = config.get("speed", {}).get("speed_limit", 80)
        self.latest_frame: Optional[bytes] = None
        self.stats = _empty_stats()
        self.lock = threading.RLock()
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def start(self, video_path: Optional[str] = None, speed_limit: Optional[int] = None):
        # Önceki akış varsa tamamen bitmesini bekle (aynı anda iki işçi çalışmasın)
        self.stop()

        with self.lock:
            path = video_path or self.current_video_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video dosyası bulunamadı: {path}")

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Video açılamadı: {path}")

            # Yalnızca doğrulanmış değerleri kalıcı yap
            self.current_video_path = path
            if speed_limit:
                self.speed_limit = int(speed_limit)

            pipeline = TrafficPipeline(config=_build_run_config(self.speed_limit),
                                       video_path=str(path))
            pipeline.violation_manager.start_session(str(self.current_video_path))

            self.pipeline = pipeline
            self.latest_frame = None
            self.stats = _empty_stats()
            self.stats["is_running"] = True
            self.is_running = True
            self.is_paused = False

            stop_event = threading.Event()
            self._stop_event = stop_event
            self.worker_thread = threading.Thread(
                target=self._process_loop, args=(cap, pipeline, stop_event), daemon=True
            )
            self.worker_thread.start()

    def _process_loop(self, cap: cv2.VideoCapture, pipeline: TrafficPipeline, stop_event: threading.Event):
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        t_start = time.time()
        checked_plates = set()

        try:
            while not stop_event.is_set():
                if self.is_paused:
                    time.sleep(0.05)
                    t_start += 0.05  # Duraklatma süresini FPS hesabına katma
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                annotated, summary = pipeline.process_frame(frame, frame_idx, fps)

                # Kara liste kontrolü (her kilitlenen plaka için bir kez)
                for trk_id, state in pipeline.plate_recognizer.track_states.items():
                    p_txt = state.locked_text
                    if not state.is_locked or not p_txt or p_txt in checked_plates:
                        continue
                    checked_plates.add(p_txt)
                    match = db.check_watchlist(p_txt)
                    if match:
                        alert_obj = {
                            "track_id": int(trk_id),
                            "plate": p_txt,
                            "reason": match.get("reason", "Aranan Araç"),
                            "severity": match.get("severity", "danger"),
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        }
                        with self.lock:
                            self.stats["watchlist_alerts"].append(alert_obj)
                            self.stats["latest_alert"] = alert_obj

                # Web yayını için optimize edilmiş JPEG sıkıştırması (4K yükünü hafifletir)
                h_frame, w_frame = annotated.shape[:2]
                if w_frame > 1280:
                    scale_factor = 1280.0 / w_frame
                    web_frame = cv2.resize(annotated, (1280, int(h_frame * scale_factor)), interpolation=cv2.INTER_LINEAR)
                else:
                    web_frame = annotated

                ok, jpeg = cv2.imencode(".jpg", web_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self.lock:
                    if ok:
                        self.latest_frame = jpeg.tobytes()
                    curr_fps = frame_idx / max(0.001, (time.time() - t_start))
                    self.stats.update({
                        "total_vehicles": summary.get("total_vehicles", 0),
                        "speed_measured_count": summary.get("speed_measured_count", 0),
                        "plate_recognized_count": summary.get("plate_recognized_count", 0),
                        "violations_count": summary.get("violations_count", 0),
                        "average_speed": summary.get("average_speed", 0.0),
                        "max_speed": summary.get("max_speed", 0),
                        "fps": round(curr_fps, 1),
                        "frame_no": frame_idx
                    })

                # Donanım nefes alması için ufak bekleme
                time.sleep(0.005)
        except Exception as e:
            print(f"[Stream] Hata: {e}")
            with self.lock:
                self.stats["error"] = str(e)
        finally:
            cap.release()
            try:
                pipeline.finalize_all()
                pipeline.violation_manager.export_csv()
            except Exception as e:
                print(f"[Stream] CSV dışa aktarma uyarısı: {e}")
            with self.lock:
                # Yalnızca hâlâ aktif olan akışsa durumu kapat
                if self._stop_event is stop_event:
                    self.is_running = False
                    self.is_paused = False
                    self.stats["is_running"] = False
                    self.stats["is_paused"] = False

    def pause(self):
        with self.lock:
            if self.is_running:
                self.is_paused = not self.is_paused
                self.stats["is_paused"] = self.is_paused

    def stop(self, wait: bool = True):
        with self.lock:
            if self._stop_event:
                self._stop_event.set()
            thread = self.worker_thread
            self.is_running = False
            self.is_paused = False
            self.stats["is_running"] = False
            self.stats["is_paused"] = False
        if wait and thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=15)

    def reset_stats(self):
        with self.lock:
            running = self.is_running
            self.stats = _empty_stats()
            self.stats["is_running"] = running


streamer = StreamController()


# ------------------------------------------------------------------------------
# Frontend & API Endpoints
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(os.path.join(LEGACY_DIR, "web"), "index.html")


@app.route("/video_feed")
def video_feed():
    """Gerçek zamanlı MJPEG video akış ucu."""
    if not streamer.is_running and streamer.latest_frame is None:
        return "", 204

    def frame_generator():
        last_sent = None
        while True:
            frame = streamer.latest_frame
            if frame is not None and frame is not last_sent:
                last_sent = frame
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            if not streamer.is_running:
                break  # Akış bitti: son kare gönderildi, bağlantıyı kapat
            time.sleep(0.033)  # ~30 FPS

    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def get_stats():
    """Anlık telemetri ve sistem sayaçları."""
    with streamer.lock:
        return jsonify(dict(streamer.stats))


@app.route("/api/config")
def get_config():
    """Arayüzün başlangıç değerleri."""
    return jsonify({
        "default_video_path": streamer.current_video_path,
        "speed_limit": streamer.speed_limit
    })


@app.route("/api/control", methods=["POST"])
def control_stream():
    """Oynat, Duraklat, Durdur komutları."""
    data = request.get_json(silent=True) or {}
    action = data.get("action")

    if action == "start":
        if batch_state["is_processing"]:
            return jsonify({"error": "Tam video analizi sürüyor. Önce onu durdurun."}), 409
        try:
            streamer.start(video_path=(data.get("video_path") or "").strip() or None,
                           speed_limit=data.get("speed_limit"))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"status": "started", "video": streamer.current_video_path})
    elif action == "pause":
        streamer.pause()
        return jsonify({"status": "paused", "is_paused": streamer.is_paused})
    elif action == "stop":
        streamer.stop()
        return jsonify({"status": "stopped"})

    return jsonify({"error": "Invalid action"}), 400


@app.route("/api/records")
def get_records():
    """Veritabanından tüm kayıtları çeker."""
    plate = request.args.get("plate", "")
    violations_only = request.args.get("violations_only", "false").lower() == "true"

    records = db.get_all_records()

    if plate:
        records = [r for r in records if plate.upper() in str(r.get("plate_text") or "").upper()]
    if violations_only:
        records = [r for r in records if r.get("is_speeding")]

    # Kara liste kontrolü ile zenginleştir
    for r in records:
        p = r.get("plate_text")
        if p:
            match = db.check_watchlist(p)
            if match:
                r["watchlist_match"] = match

    return jsonify(records)


@app.route("/api/records/<int:record_id>/pdf")
def download_record_pdf(record_id: int):
    """Araç kaydı için resmi formatta PDF İhlal Tutanağı üretir ve indirir."""
    record = db.get_record_by_id(record_id)
    if not record:
        return jsonify({"error": "Kayıt bulunamadı."}), 404

    pdf_dir = "output/raporlar"
    os.makedirs(pdf_dir, exist_ok=True)
    plate = record.get("plate_text") or f"arac_{record_id}"
    safe_plate = "".join(c for c in str(plate) if c.isalnum())
    pdf_path = os.path.join(pdf_dir, f"ihlal_tutanagi_{record_id}_{safe_plate}.pdf")

    try:
        generate_violation_pdf(record, pdf_path)
        return send_file(
            os.path.abspath(pdf_path),
            as_attachment=True,
            download_name=f"ihlal_tutanagi_{safe_plate}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"error": f"PDF oluşturma hatası: {e}"}), 500


def _record_from_snapshot_filename(filename: str) -> Optional[dict]:
    """Veritabanında kaydı olmayan kanıt fotoğrafı için dosya adından bilgi çıkarır.
    Dosya adı biçimi: violation_id{track}_{plaka}_{hız}kmh.jpg"""
    snap_path = os.path.join(os.path.abspath(VIOLATIONS_DIR), filename)
    if not os.path.isfile(snap_path):
        return None
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "violation" or not parts[1].startswith("id"):
        return None
    try:
        track_id = int(parts[1][2:])
        speed = int(parts[-1].replace("kmh", ""))
    except ValueError:
        return None
    plate = "_".join(parts[2:-1])
    return {
        "id": track_id,
        "track_id": track_id,
        "plate_text": None if plate == "UNKNOWN" else plate,
        "speed_kmh": speed,
        "speed_limit": config.get("speed", {}).get("speed_limit", 80),
        "is_speeding": True,
        "class_name": None,
        "lane_name": None,
        "timestamp": datetime.fromtimestamp(os.path.getmtime(snap_path)).strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_path": snap_path
    }


@app.route("/api/violations/<filename>/pdf")
def download_violation_image_pdf(filename: str):
    """Kanıt fotoğrafı üzerinden PDF İhlal Tutanağı üretir."""
    filename = secure_filename(filename)
    record = db.get_record_by_snapshot(filename) or _record_from_snapshot_filename(filename)
    if not record:
        return jsonify({"error": "Bu kanıt fotoğrafına ait kayıt bulunamadı."}), 404

    pdf_dir = "output/raporlar"
    os.makedirs(pdf_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]
    pdf_path = os.path.join(pdf_dir, f"tutanak_{base}.pdf")
    try:
        generate_violation_pdf(record, pdf_path)
        return send_file(
            os.path.abspath(pdf_path),
            as_attachment=True,
            download_name=f"tutanak_{base}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"error": f"PDF hatası: {e}"}), 500


# ------------------------------------------------------------------------------
# Kara Liste (Aranan Araç) API Uç Noktaları
# ------------------------------------------------------------------------------
@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    """Aranan araçlar / kara liste listesi."""
    return jsonify(db.get_watchlist())


@app.route("/api/watchlist", methods=["POST"])
def add_watchlist():
    """Kara listeye yeni plaka ekler."""
    data = request.get_json(silent=True) or {}
    plate = (data.get("plate_text") or "").strip().upper()
    reason = (data.get("reason") or "Şüpheli / Aranan Araç").strip()
    severity = data.get("severity") if data.get("severity") in ("danger", "warning") else "danger"
    if not plate:
        return jsonify({"error": "Plaka gereklidir."}), 400

    success = db.add_to_watchlist(plate, reason, severity)
    return jsonify({"status": "added" if success else "error"})


@app.route("/api/watchlist/<int:item_id>", methods=["DELETE", "POST"])
def delete_watchlist(item_id: int):
    """Kara listeden plaka kaydını kaldırır."""
    success = db.remove_from_watchlist(item_id)
    return jsonify({"status": "deleted" if success else "error"})


# ------------------------------------------------------------------------------
# Trafik Analitik İstatistikleri (Chart.js için)
# ------------------------------------------------------------------------------
@app.route("/api/stats/analytics")
def get_analytics():
    """Hız dağılımı ve araç sınıfı istatistikleri."""
    return jsonify(db.get_analytics_stats())


@app.route("/api/records/clear", methods=["POST", "DELETE"])
def clear_all_records():
    """Tüm araç ve ihlal kayıtlarını sıfırlar."""
    success = db.clear_records()
    pipeline = streamer.pipeline
    if pipeline is not None:
        pipeline.violation_manager.records.clear()
        pipeline.violation_manager.saved_snapshots.clear()
    streamer.reset_stats()
    return jsonify({"status": "cleared" if success else "error"})


@app.route("/api/records/<int:record_id>/plate", methods=["PATCH", "POST"])
def update_record_plate(record_id: int):
    """Yanlış okunan plakayı arayüzden elle düzeltme uç noktası."""
    data = request.get_json(silent=True) or {}
    new_plate = (data.get("plate_text") or "").strip().upper()
    if not new_plate:
        return jsonify({"error": "Geçerli bir plaka metni girilmelidir."}), 400

    success = db.update_plate_text(record_id, new_plate)
    if success:
        return jsonify({"status": "success", "id": record_id, "updated_plate": new_plate})
    return jsonify({"error": "Kayıt bulunamadı veya güncellenemedi."}), 404


@app.route("/api/violations")
def get_violations():
    """İhlal fotoğraflarının listesi (en yeniler önce)."""
    images = []
    if os.path.exists(VIOLATIONS_DIR):
        files = [f for f in os.listdir(VIOLATIONS_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(VIOLATIONS_DIR, f)), reverse=True)
        for f in files:
            images.append({"filename": f, "url": f"/violations_images/{f}"})
    return jsonify(images)


@app.route("/violations_images/<filename>")
def serve_violation_image(filename):
    return send_from_directory(os.path.abspath(VIOLATIONS_DIR), filename)


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Kullanıcının yüklediği videoyu kaydeder."""
    if "video" not in request.files:
        return jsonify({"error": "Dosya gönderilmedi."}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Dosya adı boş."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify({"error": f"Desteklenmeyen dosya türü: {ext or '?'} (MP4/AVI/MOV/MKV)"}), 400

    filename = secure_filename(file.filename)
    if not filename or filename.startswith("."):
        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_path = os.path.join(UPLOAD_DIR, filename)
    file.save(saved_path)

    return jsonify({"status": "uploaded", "filepath": os.path.abspath(saved_path)})


@app.route("/api/export/csv")
def export_csv():
    """Veritabanındaki tüm kayıtları CSV olarak indirir."""
    records = db.get_all_records()
    if not records:
        return jsonify({"error": "Dışa aktarılacak kayıt yok."}), 404
    csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_surveillance_records.csv"}
    )


@app.route("/api/export/db")
def export_db():
    if os.path.exists(DB_PATH):
        return send_file(os.path.abspath(DB_PATH), as_attachment=True, download_name="traffic_surveillance.db")
    return jsonify({"error": "Database not found"}), 404


# ------------------------------------------------------------------------------
# Toplu (Batch) Video İşleme
# ------------------------------------------------------------------------------
class BatchCancelled(Exception):
    pass


batch_state = {
    "is_processing": False,
    "progress": 0,
    "current_frame": 0,
    "total_frames": 0,
    "completed": False,
    "cancelled": False,
    "output_video": "/api/processed_video",
    "error": None
}
batch_cancel_event = threading.Event()


def _run_batch_processing(video_path: str, speed_limit: int):
    raw_output_path = os.path.abspath("output/temp_batch_raw.mp4")
    final_output_path = os.path.abspath("output/analiz_edilen_video.mp4")
    try:
        pipeline = TrafficPipeline(config=_build_run_config(speed_limit))
        pipeline.violation_manager.start_session(str(video_path))
        os.makedirs("output", exist_ok=True)

        def progress_cb(current, total, summary):
            if batch_cancel_event.is_set():
                raise BatchCancelled()
            batch_state["current_frame"] = current
            batch_state["total_frames"] = total
            if total > 0:
                batch_state["progress"] = min(99, int((current / total) * 100))
            # Üstteki KPI kartları toplu analiz sırasında da güncellensin
            with streamer.lock:
                streamer.stats.update({k: summary.get(k, streamer.stats.get(k)) for k in (
                    "total_vehicles", "speed_measured_count", "plate_recognized_count",
                    "violations_count", "average_speed", "max_speed")})
                streamer.stats["frame_no"] = current

        pipeline.process_video(
            video_path=video_path,
            output_path=raw_output_path,
            show_window=False,
            progress_callback=progress_cb
        )

        # H.264 dönüştürme (Tüm tarayıcılarda akıcı HTML5 oynatımı için)
        converted = False
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y", "-i", raw_output_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-movflags", "+faststart", final_output_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            converted = result.returncode == 0 and os.path.exists(final_output_path)
        except Exception as fe:
            print(f"[FFmpeg] Dönüştürme uyarısı: {fe}")

        if converted:
            try:
                os.remove(raw_output_path)
            except OSError:
                pass
        elif os.path.exists(raw_output_path):
            os.replace(raw_output_path, final_output_path)

        batch_state["progress"] = 100
        batch_state["completed"] = True

    except BatchCancelled:
        batch_state["cancelled"] = True
    except Exception as e:
        print(f"[Batch] Hata: {e}")
        batch_state["error"] = str(e)
    finally:
        batch_state["is_processing"] = False


@app.route("/api/batch/start", methods=["POST"])
def start_batch():
    """Videoyu canlı yayınlamadan arka planda yüksek hızda baştan sona işler."""
    data = request.get_json(silent=True) or {}
    video_path = (data.get("video_path") or "").strip() or streamer.current_video_path
    try:
        speed_limit = int(data.get("speed_limit") or streamer.speed_limit)
    except (TypeError, ValueError):
        speed_limit = streamer.speed_limit

    if batch_state["is_processing"]:
        return jsonify({"status": "already_running"}), 400
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video dosyası bulunamadı: {video_path}"}), 400

    # GPU'yu paylaşmamak için canlı akışı durdur
    streamer.stop()
    streamer.reset_stats()

    batch_cancel_event.clear()
    batch_state.update({
        "is_processing": True,
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "completed": False,
        "cancelled": False,
        "error": None
    })
    threading.Thread(target=_run_batch_processing, args=(video_path, speed_limit), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/batch/cancel", methods=["POST"])
def cancel_batch():
    """Devam eden toplu analizi iptal eder."""
    if batch_state["is_processing"]:
        batch_cancel_event.set()
        return jsonify({"status": "cancelling"})
    return jsonify({"status": "idle"})


@app.route("/api/batch/status")
def get_batch_status():
    """Toplu video işleme ilerleme durumu."""
    return jsonify(batch_state)


@app.route("/api/processed_video")
def get_processed_video():
    """Tamamlanmış, üzerinde hız ve plaka bilgileri işlenmiş videoyu sunar."""
    path = os.path.abspath("output/analiz_edilen_video.mp4")
    if os.path.exists(path):
        return send_file(path, mimetype="video/mp4", conditional=True)
    return jsonify({"error": "İşlenmiş video henüz hazır değil."}), 404


if __name__ == "__main__":
    print("🚀 SMART TRAFFIC PRO Web Sunucusu Başlatılıyor: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
