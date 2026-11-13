"""
UA-DETRAC (veya herhangi bir kare dizisi) -> MP4 demo videosu.
Hocanın şartı: final kabul testi açık bir veri setinden videolarla yapılır.

  python scripts/frames_to_video.py --frames D:/UA-DETRAC/Insight-MVT_Annotation_Test/MVI_39031 --out output/uploads/MVI_39031.mp4
"""
import argparse
import glob
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Kare klasörü (img00001.jpg …)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0, help="UA-DETRAC 25 fps'tir")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) + glob.glob(os.path.join(args.frames, "*.png")))
    if not files:
        raise SystemExit("Kare bulunamadı.")
    h, w = cv2.imread(files[0]).shape[:2]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in files:
        vw.write(cv2.imread(f))
    vw.release()
    print(f"✅ {len(files)} kare -> {args.out}")


if __name__ == "__main__":
    main()
