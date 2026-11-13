"""
FR-06 / NFR-2: Plaka okuma değerlendirmesi (FPA = tam plaka doğruluğu)

Test seti düzeni (docs/DATASETS.md):
  A) Tek görüntü:   <dir>/labels.csv  (filename,plate)   veya dosya adı = plaka (34ABC123.jpg, 34ABC123_2.jpg)
  B) İz (track):    <dir>/<PLAKA>/*.jpg   — aynı aracın ardışık kareleri (zamansal oylama testi)

Karşılaştırılan yöntemler (RSA Tablo 2):
  (a) easyocr      : EasyOCR, plaka kırpıntısı üzerinde (kuruluysa)
  (b) single       : FastALPR tek kare, ham çıktı
  (c) proposed     : FastALPR + TR sözdizimi doğrulama/düzeltme (+ B düzeninde zamansal oylama)

Metrikler: FPA (tüm karakterler doğru), CRA (karakter doğruluğu), misread (yanlış ama boş değil),
reject (okunamadı).  Çıktı: output/eval/alpr_<method>.json
"""
import argparse
import csv
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import levenshtein, write_result  # noqa: E402
from src.plate_recognizer import PlateCandidate, PlateRecognizer, PlateTrackState  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def norm(t: str) -> str:
    return "".join(c for c in (t or "").upper() if c.isalnum())


def load_items(root: str):
    """[(label, [image_paths])] döndürür."""
    items = []
    lab_csv = os.path.join(root, "labels.csv")
    if os.path.exists(lab_csv):
        with open(lab_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                items.append((norm(r["plate"]), [os.path.join(root, r["filename"])]))
        return items
    subdirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if subdirs:
        for d in subdirs:
            imgs = sorted(p for p in glob.glob(os.path.join(root, d, "*")) if p.lower().endswith(IMG_EXT))
            if imgs:
                items.append((norm(d), imgs))
        return items
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        if p.lower().endswith(IMG_EXT):
            items.append((norm(os.path.splitext(os.path.basename(p))[0].split("_")[0]), [p]))
    return items


def read_fastalpr(alpr, img):
    """(ham_metin, ortalama_güven, karakter_güvenleri) — en güvenli plaka."""
    best = None
    for r in alpr.predict(img) or []:
        if getattr(r, "ocr", None) is None or not r.ocr.text:
            continue
        mean_c, cc = PlateRecognizer._ocr_confidences(r.ocr.confidence)
        if best is None or mean_c > best[1]:
            best = (norm(r.ocr.text), mean_c, cc)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Türk plaka test seti klasörü")
    ap.add_argument("--method", choices=["single", "proposed", "easyocr"], default="proposed")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = load_items(args.data)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("Test setinde görüntü bulunamadı.")

    rec = PlateRecognizer(plate_format="tr", init_alpr=args.method != "easyocr")
    reader = None
    if args.method == "easyocr":
        import easyocr  # pip install easyocr
        reader = easyocr.Reader(["en"], gpu=False)
    elif rec.alpr is None:
        raise SystemExit("fast-alpr yüklenemedi.")

    n_ok = n_reject = n_mis = 0
    char_err = char_tot = 0
    per = []
    for label, paths in items:
        pred = None
        if args.method == "easyocr":
            img = cv2.imread(paths[0])
            res = reader.readtext(img, detail=1, allowlist="ABCDEFGHIJKLMNOPRSTUVYZ0123456789")
            if res:
                pred = norm("".join(t for _, t, _ in sorted(res, key=lambda x: x[0][0][0])))
        elif args.method == "single":
            r = read_fastalpr(rec.alpr, cv2.imread(paths[0]))
            pred = r[0] if r else None
        else:
            st = PlateTrackState()
            for p in paths:
                r = read_fastalpr(rec.alpr, cv2.imread(p))
                if r is None:
                    continue
                clean = rec.clean_read(r[0], r[2] if r[2] and len(r[2]) == len(r[0]) else None)
                if clean and r[1] >= rec.min_ocr_conf:
                    st.candidates.append(PlateCandidate(clean, r[1], None))
            if len(paths) == 1 and st.candidates:
                pred = st.candidates[0].text            # tek kare: doğrulanmış/düzeltilmiş okuma
            else:
                rec._try_lock(st, final=True)
                pred = st.locked_text
        if not pred:
            n_reject += 1
        elif pred == label:
            n_ok += 1
        else:
            n_mis += 1
        char_err += levenshtein(pred or "", label)
        char_tot += len(label)
        per.append({"label": label, "pred": pred, "n_frames": len(paths)})

    n = len(items)
    metrics = {
        "FPA_pct": round(100.0 * n_ok / n, 2),
        "CRA_pct": round(100.0 * max(0, 1 - char_err / max(1, char_tot)), 2),
        "misread_pct": round(100.0 * n_mis / n, 2),
        "reject_pct": round(100.0 * n_reject / n, 2),
        "samples": n,
        "NFR2_target_FPA_ge_90": n_ok / n >= 0.90,
    }
    write_result(f"alpr_{args.method}", metrics, {"data": os.path.abspath(args.data), "method": args.method}, per)


if __name__ == "__main__":
    main()
