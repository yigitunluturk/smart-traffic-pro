import os
import threading
import time
from typing import Optional
import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory
from flask_cors import CORS
import yaml

from src.pipeline import TrafficPipeline
from src.database import TrafficDatabase

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

# Global Durum
CONFIG_PATH = "config/default_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db = TrafficDatabase(db_path=config.get("output", {}).get("database_path", "output/traffic_surveillance.db"))

class StreamController:
    def __init__(self):
        self.pipeline: Optional[TrafficPipeline] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False
        self.current_video_path = r"C:\Users\msi-nb\Desktop\part1.mp4" if os.path.exists(r"C:\Users\msi-nb\Desktop\part1.mp4") else "video.mp4"
        self.speed_limit = config.get("speed", {}).get("speed_limit", 80)
        self.latest_frame: Optional[bytes] = None
        self.stats = {
            "total_vehicles": 0,
            "speed_measured_count": 0,
            "plate_recognized_count": 0,
            "violations_count": 0,
            "average_speed": 0.0,
            "max_speed": 0,
            "fps": 0.0,
            "frame_no": 0,
            "is_running": False
        }
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None

    def start(self, video_path: Optional[str] = None, speed_limit: Optional[int] = None):
        with self.lock:
            if self.is_running:
                self.stop()

            if video_path:
                self.current_video_path = video_path
            if speed_limit:
                self.speed_limit = speed_limit

            # Konfigürasyonu hazırla
            run_cfg = config.copy()
            run_cfg["speed"]["speed_limit"] = self.speed_limit
            
            self.pipeline = TrafficPipeline(config=run_cfg)
            self.pipeline.violation_manager.start_session(str(self.current_video_path))
            
            self.cap = cv2.VideoCapture(self.current_video_path)
            self.is_running = True
            self.is_paused = False
            self.stats["is_running"] = True

            self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.worker_thread.start()

    def _process_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        t_start = time.time()

        while self.is_running and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated, summary = self.pipeline.process_frame(frame, frame_idx, fps)

            # Web yayını için optimize edilmiş JPEG sıkıştırması
            ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.lock:
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

        self.stop()

    def pause(self):
        with self.lock:
            self.is_paused = not self.is_paused

    def stop(self):
        with self.lock:
            self.is_running = False
            self.is_paused = False
            self.stats["is_running"] = False
            if self.cap:
                self.cap.release()
            if self.pipeline:
                self.pipeline.violation_manager.export_csv()


streamer = StreamController()


# ------------------------------------------------------------------------------
# Frontend & API Endpoints
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/video_feed")
def video_feed():
    """Gerçek zamanlı MJPEG video akış ucu."""
    def frame_generator():
        while True:
            if streamer.latest_frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + streamer.latest_frame + b"\r\n")
            time.sleep(0.033)  # ~30 FPS

    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def get_stats():
    """Anlık telemetri ve sistem sayaçları."""
    return jsonify(streamer.stats)


@app.route("/api/control", methods=["POST"])
def control_stream():
    """Oynat, Duraklat, Durdur komutları."""
    data = request.json or {}
    action = data.get("action")

    if action == "start":
        video_path = data.get("video_path")
        speed_limit = data.get("speed_limit")
        streamer.start(video_path=video_path, speed_limit=speed_limit)
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
        records = [r for r in records if plate.upper() in str(r.get("plate_text", "")).upper()]
    if violations_only:
        records = [r for r in records if r.get("is_speeding")]
        
    return jsonify(records)


@app.route("/api/records/<int:record_id>/plate", methods=["PATCH", "POST"])
def update_record_plate(record_id: int):
    """Yanlış okunan plakayı arayüzden elle düzeltme uç noktası."""
    data = request.json or {}
    new_plate = data.get("plate_text", "").strip().upper()
    if not new_plate:
        return jsonify({"error": "Geçerli bir plaka metni girilmelidir."}), 400
    
    success = db.update_plate_text(record_id, new_plate)
    if success:
        return jsonify({"status": "success", "id": record_id, "updated_plate": new_plate})
    return jsonify({"error": "Kayıt bulunamadı veya güncellenemedi."}), 404


@app.route("/api/violations")
def get_violations():
    """İhlal fotoğraflarının listesi."""
    viol_dir = config.get("output", {}).get("violations_dir", "output/ihlaller")
    images = []
    if os.path.exists(viol_dir):
        for f in os.listdir(viol_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                images.append({
                    "filename": f,
                    "url": f"/violations_images/{f}"
                })
    return jsonify(images)


@app.route("/violations_images/<filename>")
def serve_violation_image(filename):
    viol_dir = os.path.abspath(config.get("output", {}).get("violations_dir", "output/ihlaller"))
    return send_from_directory(viol_dir, filename)


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Kullanıcının yüklediği videoyu kaydeder."""
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_dir = "output/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, file.filename)
    file.save(saved_path)

    return jsonify({"status": "uploaded", "filepath": os.path.abspath(saved_path)})


@app.route("/api/export/csv")
def export_csv():
    csv_path = config.get("output", {}).get("csv_path", "output/arac_kayitlari.csv")
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name="traffic_surveillance_records.csv")
    return jsonify({"error": "CSV not found"}), 404


@app.route("/api/export/db")
def export_db():
    db_path = config.get("output", {}).get("database_path", "output/traffic_surveillance.db")
    if os.path.exists(db_path):
        return send_file(db_path, as_attachment=True, download_name="traffic_surveillance.db")
    return jsonify({"error": "Database not found"}), 404


if __name__ == "__main__":
    print("🚀 ArgusVision Cyber-SOC Web Server başlatılıyor: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
