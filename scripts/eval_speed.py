"""
FR-06 / NFR-1: Hız tahmini değerlendirmesi (MAE, km/h)

İki tür etiket desteklenir:
  1) BrnoCompSpeed oturumu:  --brno <session_dir>
       <session_dir>/video.avi  ve  <session_dir>/gt_data.pkl
       Kalibrasyon (vp1, vp2, pp, scale) gt_data.pkl'den otomatik homografiye çevrilir.
  2) Genel CSV:              --video v.mp4 --gt-csv gt.csv
       gt.csv sütunları: time_s,speed_kmh   (aracın ölçüm bölgesinden geçtiği an)
       Videonun kalibrasyonu config/calibrations/<video>.json'dan okunur.

Baseline karşılaştırması (RSA Tablo 2):
  --method homography   (önerilen: yol düzlemi kalibrasyonu)
  --method corridor     (baseline: perspektif kalibrasyonsuz naif piksel yer değiştirmesi)

Çıktı: output/eval/speed_<isim>_<method>.json
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, write_result  # noqa: E402


def brno_homography(gt: dict, frame_w: int, frame_h: int) -> dict:
    """BrnoCompSpeed kalibrasyonunu (iki kaçış noktası + ölçek) 4 noktalı homografiye çevirir."""
    cal = gt["camera_calibration"]
    vp1, vp2 = np.array(cal["vp1"], float), np.array(cal["vp2"], float)
    pp = np.array(cal.get("pp", [frame_w / 2.0, frame_h / 2.0]), float)
    scale = float(cal["scale"])
    f = np.sqrt(max(1e-6, -np.dot(vp1 - pp, vp2 - pp)))
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    d1 = Kinv @ np.array([*vp1, 1.0]); d1 /= np.linalg.norm(d1)
    d2 = Kinv @ np.array([*vp2, 1.0]); d2 /= np.linalg.norm(d2)
    n = np.cross(d1, d2); n /= np.linalg.norm(n)
    e2 = np.cross(n, d1)

    def to_plane(p):
        r = Kinv @ np.array([p[0], p[1], 1.0])
        X = r / np.dot(n, r)
        return np.array([np.dot(X, d1), np.dot(X, e2)]) * scale

    img_pts = [[0.3 * frame_w, 0.9 * frame_h], [0.7 * frame_w, 0.9 * frame_h],
               [0.7 * frame_w, 0.55 * frame_h], [0.3 * frame_w, 0.55 * frame_h]]
    world = [to_plane(p).tolist() for p in img_pts]
    base = [[p[0] * 1280.0 / frame_w, p[1] * 720.0 / frame_h] for p in img_pts]
    return {"base_resolution": [1280, 720], "image_points": base, "world_points": world,
            "measure_y_range": [0.25, 0.99]}


def load_brno(session_dir: str):
    with open(os.path.join(session_dir, "gt_data.pkl"), "rb") as f:
        gt = pickle.load(f, encoding="latin-1")
    fps = float(gt.get("fps", 50.0))
    cars = []
    for c in gt["cars"]:
        if not c.get("valid", True):
            continue
        times = [t for t in c.get("intersections", [])]
        t = float(np.mean([x["videoTime"] if isinstance(x, dict) else x for x in times])) if times else None
        if t is not None:
            cars.append({"time_s": t, "speed_kmh": float(c["speed"])})
    video = next((os.path.join(session_dir, v) for v in ("video.avi", "video.mp4") if os.path.exists(os.path.join(session_dir, v))), None)
    return video, gt, cars, fps


def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return [{"time_s": float(r["time_s"]), "speed_kmh": float(r["speed_kmh"])} for r in csv.DictReader(f)]


def run_pipeline(video: str, method: str, max_frames: int, calib: dict = None, stride: int = 1):
    import cv2
    import yaml
    from src.pipeline import TrafficPipeline

    with open(os.path.join(ROOT, "config", "default_config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["speed"]["method"] = "homography" if (method == "homography" and calib) else ("corridor" if method == "corridor" else "auto")
    if calib:
        cfg["speed"]["homography"] = calib
    cfg["model"]["frame_stride"] = stride
    cfg["output"]["save_violation_snapshots"] = False
    cfg["output"]["database_path"] = os.path.join(ROOT, "output", "eval", "eval_tmp.db")
    cfg["output"]["csv_path"] = os.path.join(ROOT, "output", "eval", "eval_tmp.csv")
    pipe = TrafficPipeline(cfg, video_path=video if method == "homography" and not calib else None)
    if method == "corridor":
        pipe.speed_estimator.H = None   # baseline: perspektif kalibrasyonu yok

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and k >= max_frames):
            break
        k += 1
        pipe.process_frame(frame, k, fps)
        if k % 500 == 0:
            print(f"  {k} kare işlendi…")
    cap.release()
    pipe.finalize_all()
    preds = []
    for tid, info in pipe.speed_estimator.results.items():
        if info.speed_kmh is not None and info.entry_frame is not None:
            t = (info.entry_frame + info.exit_frame) / 2.0 / fps
            preds.append({"track_id": tid, "time_s": t, "speed_kmh": float(info.speed_kmh)})
    return preds, fps, k


def match(gt, preds, max_dt: float):
    """Zaman olarak en yakın tahminle açgözlü bire-bir eşleştirme."""
    pairs, used = [], set()
    for g in sorted(gt, key=lambda x: x["time_s"]):
        best, best_dt = None, max_dt
        for i, p in enumerate(preds):
            if i in used:
                continue
            dt = abs(p["time_s"] - g["time_s"])
            if dt <= best_dt:
                best, best_dt = i, dt
        if best is not None:
            used.add(best)
            pairs.append((g, preds[best]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brno", help="BrnoCompSpeed oturum klasörü (video.avi + gt_data.pkl)")
    ap.add_argument("--video")
    ap.add_argument("--gt-csv")
    ap.add_argument("--method", choices=["homography", "corridor"], default="homography")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tüm video")
    ap.add_argument("--max-dt", type=float, default=1.0, help="Eşleştirme toleransı (s)")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    calib = None
    if args.brno:
        import cv2
        video, gt_raw, gt, _ = load_brno(args.brno)
        cap = cv2.VideoCapture(video)
        w, h = int(cap.get(3)), int(cap.get(4))
        cap.release()
        calib = brno_homography(gt_raw, w, h)
        name = os.path.basename(os.path.normpath(args.brno))
    elif args.video and args.gt_csv:
        video, gt = args.video, load_csv(args.gt_csv)
        name = os.path.splitext(os.path.basename(video))[0]
    else:
        ap.error("--brno veya (--video ve --gt-csv) verilmeli")

    preds, fps, n_frames = run_pipeline(video, args.method, args.max_frames, calib, args.stride)
    t_end = n_frames / fps
    gt = [g for g in gt if g["time_s"] <= t_end]
    pairs = match(gt, preds, args.max_dt)
    err = np.array([p["speed_kmh"] - g["speed_kmh"] for g, p in pairs]) if pairs else np.array([])

    metrics = {
        "MAE_kmh": round(float(np.mean(np.abs(err))), 3) if len(err) else None,
        "RMSE_kmh": round(float(np.sqrt(np.mean(err ** 2))), 3) if len(err) else None,
        "median_abs_err_kmh": round(float(np.median(np.abs(err))), 3) if len(err) else None,
        "mean_rel_err_pct": round(float(np.mean(np.abs(err) / np.array([g["speed_kmh"] for g, _ in pairs]) * 100)), 2) if len(err) else None,
        "gt_vehicles": len(gt),
        "predicted_vehicles": len(preds),
        "matched": len(pairs),
        "recall_pct": round(100.0 * len(pairs) / max(1, len(gt)), 1),
        "NFR1_target_MAE_le_2": (bool(np.mean(np.abs(err)) <= 2.0) if len(err) else None),
    }
    per = [{"gt_time": g["time_s"], "gt": g["speed_kmh"], "pred": p["speed_kmh"], "track": p["track_id"]} for g, p in pairs]
    write_result(f"speed_{name}_{args.method}", metrics,
                 {"video": video, "method": args.method, "frames": n_frames, "fps": fps,
                  "max_dt": args.max_dt, "stride": args.stride}, per)


if __name__ == "__main__":
    main()
