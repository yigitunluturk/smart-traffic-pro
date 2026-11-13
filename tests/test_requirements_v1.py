"""
RSA gereksinimleri için testler:
  FR-07/NFR-5 gizlilik ve saklama, FR-08 filtreleme, FR-09 doğrulamalı düzeltme + log,
  FR-06 değerlendirme betiklerinin yardımcı fonksiyonları, uçtan uca pipeline senaryosu.
"""
import os
import sys
import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.database import TrafficDatabase
from src.plate_recognizer import is_valid_tr
from src.privacy import PlatePrivacy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))


@pytest.fixture
def db():
    p = f"output/test_{uuid.uuid4().hex[:8]}.db"
    d = TrafficDatabase(db_path=p)
    yield d
    try:
        os.remove(p)
    except OSError:
        pass


def _rec(tid, plate=None, speeding=False, ts=None, lane="Serit 1", cand=None):
    return {"track_id": tid, "class_name": "car", "speed_kmh": 95 if speeding else 50, "speed_limit": 80,
            "is_speeding": speeding, "plate_text": plate, "plate_confidence": 0.9 if plate else None,
            "lane_name": lane, "elapsed_time_sec": 1.2, "first_seen_frame": 1, "last_seen_frame": 5,
            "timestamp": ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "snapshot_path": None,
            "plate_candidate": cand, "plate_status": "valid" if plate else ("low_confidence" if cand else None)}


def test_fr09_correction_validated_and_logged(db):
    sid = db.create_session("v.mp4", 80)
    db.save_or_update_vehicle(_rec(1, cand="34A8C123", speeding=True), session_id=sid)
    rid = db.query_records()[0]["id"]
    ok, _ = db.correct_plate(rid, "34 QBC 123", validator=is_valid_tr)
    assert not ok                                    # Q kullanılmaz -> reddedilir
    ok, _ = db.correct_plate(rid, "34 ABC 123", validator=is_valid_tr)
    assert ok
    r = db.get_record_by_id(rid)
    assert r["plate_text"] == "34ABC123" and r["plate_status"] == "corrected"
    log = db.get_corrections(rid)
    assert log[0]["old_plate"] == "34A8C123" and log[0]["new_plate"] == "34ABC123" and log[0]["corrected_at"]


def test_fr08_filters(db):
    old = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    db.save_or_update_vehicle(_rec(1, plate="34ABC123", lane="A", ts=old))
    db.save_or_update_vehicle(_rec(2, plate="06XY999", lane="B", speeding=True))
    db.save_or_update_vehicle(_rec(3, cand="34KLM55", lane="B"))
    assert len(db.query_records(plate_query="34")) == 2           # kesinleşmiş + aday
    assert len(db.query_records(lane="B")) == 2
    assert len(db.query_records(violations_only=True)) == 1
    today = datetime.now().strftime("%Y-%m-%d")
    assert len(db.query_records(date_from=today, date_to=today)) == 2
    assert set(db.get_lanes()) == {"A", "B"}


def test_nfr5_retention_keeps_violations(db):
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    db.save_or_update_vehicle(_rec(1, plate="34ABC123", ts=old))
    db.save_or_update_vehicle(_rec(2, plate="06XY999", ts=old, speeding=True))
    db.save_or_update_vehicle(_rec(3, plate="35AB123"))
    assert db.apply_retention(30) == 1
    recs = {r["track_id"]: r for r in db.query_records()}
    assert recs[1]["plate_text"] is None and recs[1]["plate_status"] == "purged"
    assert recs[2]["plate_text"] == "06XY999"     # ihlal kanıtı korunur
    assert recs[3]["plate_text"] == "35AB123"     # süresi dolmamış


def test_fr07_blur_changes_only_plate_region():
    rng = np.random.default_rng(0)
    frame = (rng.random((400, 600, 3)) * 255).astype(np.uint8)
    orig = frame.copy()
    pp = PlatePrivacy()
    box = (100, 100, 300, 260)
    pp.observe_plate(7, box, (170, 230, 230, 250))
    region = pp.region_for(7, box, frame.shape)
    pp.blur(frame, region)
    x1, y1, x2, y2 = region
    assert x1 >= 100 and x2 <= 300 and y1 >= 200
    assert not np.array_equal(frame[y1:y2, x1:x2], orig[y1:y2, x1:x2])
    mask = np.ones(frame.shape[:2], bool)
    mask[y1:y2, x1:x2] = False
    assert np.array_equal(frame[mask], orig[mask])


def test_eval_helpers():
    from _common import levenshtein
    from eval_speed import match
    assert levenshtein("34ABC123", "34A8C123") == 1
    gt = [{"time_s": 1.0, "speed_kmh": 50}, {"time_s": 3.0, "speed_kmh": 70}]
    pr = [{"time_s": 3.2, "speed_kmh": 71, "track_id": 2}, {"time_s": 1.1, "speed_kmh": 49, "track_id": 1}]
    pairs = match(gt, pr, max_dt=0.5)
    assert [(g["speed_kmh"], p["track_id"]) for g, p in pairs] == [(50, 1), (70, 2)]


def test_brno_calibration_conversion_scale():
    """Sentetik kamera: iki kaçış noktası + ölçekten üretilen homografi metre mesafeyi korumalı."""
    from eval_speed import brno_homography
    import cv2
    f, pp = 1000.0, np.array([960.0, 540.0])
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])
    yaw, pit, cam_h = np.deg2rad(25), np.deg2rad(18), 8.0
    fc = np.array([np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), -np.sin(pit)])
    rc = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    dc = np.cross(fc, rc)
    R = np.vstack([rc, dc, fc])
    C = np.array([0, 0, cam_h])

    def proj(Xw):
        p = K @ (R @ (np.array(Xw, float) - C))
        return p[:2] / p[2]

    def vp(u):
        p = K @ (R @ np.array(u, float))
        return p[:2] / p[2]
    vp1, vp2 = vp([0, 1, 0]), vp([1, 0, 0])
    # 'scale' Brno'da: düzlem birimi -> metre. Düzlem n·X=1 kamera yüksekliği h'ye karşılık gelir -> scale = h
    gt = {"camera_calibration": {"vp1": vp1.tolist(), "vp2": vp2.tolist(), "pp": pp.tolist(), "scale": cam_h}}
    cal = brno_homography(gt, 1920, 1080)
    H, _ = cv2.findHomography(np.float32(cal["image_points"]), np.float32(cal["world_points"]))
    a, b = proj([0, 20, 0]), proj([0, 30, 0])
    A = cv2.perspectiveTransform(np.float32([[[a[0] * 1280 / 1920, a[1] * 720 / 1080]]]), H)[0, 0]
    B = cv2.perspectiveTransform(np.float32([[[b[0] * 1280 / 1920, b[1] * 720 / 1080]]]), H)[0, 0]
    assert abs(np.linalg.norm(A - B) - 10.0) < 0.3


class _Det:
    def __init__(self, tid, box):
        self.track_id, self.box = tid, box
        self.center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        self.width, self.height = box[2] - box[0], box[3] - box[1]
        self.confidence, self.class_id, self.class_name = 0.9, 2, "car"


def test_pipeline_end_to_end_speeding_vehicle(tmp_path):
    """Uçtan uca: kalibre sahnede hızlı giden araç -> hız kilitlenir, ihlal kaydı + kanıt fotoğrafı oluşur."""
    import yaml
    from src.pipeline import TrafficPipeline
    with open("config/default_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["speed"]["method"] = "homography"
    cfg["speed"]["homography"] = {
        "base_resolution": [1280, 720],
        "image_points": [[467.4, 613.0], [635.4, 613.0], [659.5, 388.0], [586.9, 388.0]],
        "world_points": [[0.0, 0.0], [3.5, 0.0], [3.5, 24.0], [0.0, 24.0]],
        "measure_y_range": [0.3, 0.99]}
    cfg["speed"]["stabilization"] = {"enabled": False}
    cfg["output"].update({"database_path": str(tmp_path / "t.db"), "csv_path": str(tmp_path / "t.csv"),
                          "violations_dir": str(tmp_path / "viol")})
    pipe = TrafficPipeline(cfg)
    est = pipe.speed_estimator
    est.adapt_to_frame(1280, 720)
    v = 110 / 3.6
    frames = []
    for k in range(1, 70):
        p = est.world_to_image(np.array([[1.75, -5 + v * k / 30.0]]), (1280, 720))[0]
        frames.append((k, (int(p[0] - 40), int(p[1] - 60), int(p[0] + 40), int(p[1]))))
    seq = iter(frames)
    cur = {}
    pipe.detector.detect_and_track = lambda fr: [_Det(5, cur["box"])] if cur.get("box") else []
    img = np.full((720, 1280, 3), 90, np.uint8)
    for k, box in frames:
        cur["box"] = box if box[3] < 715 and box[1] > 0 else None
        pipe.process_frame(img.copy(), k, 30.0)
    cur["box"] = None
    for k in range(70, 110):
        pipe.process_frame(img.copy(), k, 30.0)
    rec = pipe.violation_manager.records[5]
    assert rec.speed_kmh is not None and abs(rec.speed_kmh - 110) <= 2
    assert rec.is_speeding and rec.snapshot_path and os.path.exists(rec.snapshot_path)
    assert "OKUNAMADI" in rec.snapshot_path       # plaka okunamadıysa yanlış plaka uydurulmaz


def test_evidence_includes_plate_crop(tmp_path):
    """UC-03: ihlal kanıtıyla birlikte plaka kırpıntısı da kaydedilir."""
    from src.violation_manager import ViolationManager
    vm = ViolationManager(speed_limit=80, csv_path=str(tmp_path / "r.csv"),
                          violations_dir=str(tmp_path / "v"), db_path=str(tmp_path / "d.db"))
    frame = np.zeros((720, 1280, 3), np.uint8)
    rec = vm.register_or_update(track_id=3, class_name="car", frame_no=5, speed_kmh=101,
                                plate_text="34ABC123", plate_conf=0.9, frame_for_snapshot=frame,
                                vehicle_box=(400, 300, 600, 460), plate_rel_box=(0.35, 0.75, 0.65, 0.9))
    assert rec.snapshot_path and os.path.exists(rec.snapshot_path[:-4] + "_plaka.jpg")
