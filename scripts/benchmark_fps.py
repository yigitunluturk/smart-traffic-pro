"""
NFR-3: CPU işlem hızı ölçümü (imgsz=640, ayarlanabilir kare atlama)

  python scripts/benchmark_fps.py --video output/uploads/x.mp4 --frames 600 --stride 2

Çıktı: output/eval/throughput_stride<N>.json  (CPU modeli ve çekirdek sayısı dahil)
"""
import argparse
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, write_result  # noqa: E402
from src.pipeline import TrafficPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config", "default_config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"].update({"device": "cpu", "imgsz": args.imgsz, "frame_stride": args.stride})
    cfg["output"]["database_path"] = os.path.join(ROOT, "output", "eval", "bench_tmp.db")
    cfg["output"]["csv_path"] = os.path.join(ROOT, "output", "eval", "bench_tmp.csv")
    cfg["output"]["save_violation_snapshots"] = False
    pipe = TrafficPipeline(cfg, video_path=args.video)

    cap = cv2.VideoCapture(args.video)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    n, t0 = 0, None
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n == 6:
            t0 = time.perf_counter()   # ilk kareler ısınma (model yükleme) — ölçüme katılmaz
        pipe.process_frame(frame, n, fps_src)
    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    cap.release()
    fps = (n - 5) / elapsed if elapsed > 0 else 0.0
    write_result(f"throughput_stride{args.stride}",
                 {"pipeline_fps": round(fps, 2), "frames": n, "NFR3_target_ge_10fps": fps >= 10},
                 {"video": args.video, "resolution": f"{w}x{h}", "imgsz": args.imgsz, "stride": args.stride})


if __name__ == "__main__":
    main()
