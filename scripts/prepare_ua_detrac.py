"""
FR-03: UA-DETRAC -> YOLO formatı (sabit, tekrarlanabilir bölme)

Girdi (resmi UA-DETRAC düzeni):
  <root>/Insight-MVT_Annotation_Train/MVI_xxxxx/img00001.jpg
  <root>/DETRAC-Train-Annotations-XML/MVI_xxxxx.xml
  (varsa) <root>/Insight-MVT_Annotation_Test + DETRAC-Test-Annotations-XML

Sınıflar COCO kimlikleriyle eşlenir, böylece pipeline config'i değişmeden çalışır:
  car -> 2 (car), van -> 2 (car), bus -> 5 (bus), others -> 7 (truck)

Bölme: eğitim dizileri seed=0 ile %85 train / %15 val (DİZİ bazında, sızıntı yok);
resmi test dizileri -> test. Bölme listesi data/ua_detrac/split.json'a yazılır.
Yoksay bölgeleri (ignored_region) gri ile boyanır.

Kullanım:
  python scripts/prepare_ua_detrac.py --root D:/UA-DETRAC --out data/ua_detrac --every 10
"""
import argparse
import glob
import json
import os
import random
import xml.etree.ElementTree as ET

import cv2

CLASS_MAP = {"car": 2, "van": 2, "bus": 5, "others": 7}
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck"}


def convert_sequence(xml_path, img_dir, out_img, out_lbl, every):
    root = ET.parse(xml_path).getroot()
    seq = os.path.splitext(os.path.basename(xml_path))[0]
    ignored = []
    ig = root.find("ignored_region")
    if ig is not None:
        for b in ig.findall("box"):
            l, t, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            ignored.append((int(l), int(t), int(l + w), int(t + h)))
    n = 0
    for fr in root.findall("frame"):
        num = int(fr.get("num"))
        if (num - 1) % every:
            continue
        src = os.path.join(img_dir, seq, f"img{num:05d}.jpg")
        img = cv2.imread(src)
        if img is None:
            continue
        H, W = img.shape[:2]
        for (x1, y1, x2, y2) in ignored:
            img[max(0, y1):y2, max(0, x1):x2] = 114
        lines = []
        tl = fr.find("target_list")
        for tg in (tl.findall("target") if tl is not None else []):
            b, a = tg.find("box"), tg.find("attribute")
            cls = CLASS_MAP.get((a.get("vehicle_type") if a is not None else "car"), 2)
            l, t, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            cx, cy = (l + w / 2) / W, (t + h / 2) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        name = f"{seq}_{num:05d}"
        cv2.imwrite(os.path.join(out_img, name + ".jpg"), img)
        with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/ua_detrac")
    ap.add_argument("--every", type=int, default=10, help="Her N karede bir örnek (ardışık kareler neredeyse aynı)")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr_xml = sorted(glob.glob(os.path.join(args.root, "DETRAC-Train-Annotations-XML", "*.xml")))
    te_xml = sorted(glob.glob(os.path.join(args.root, "DETRAC-Test-Annotations-XML", "*.xml")))
    if not tr_xml:
        raise SystemExit("DETRAC-Train-Annotations-XML bulunamadı.")
    rnd = random.Random(args.seed)
    seqs = tr_xml[:]
    rnd.shuffle(seqs)
    n_val = max(1, int(len(seqs) * args.val_ratio))
    split = {"val": sorted(seqs[:n_val]), "train": sorted(seqs[n_val:]), "test": te_xml}
    if not te_xml:   # resmi test etiketi yoksa val'in yarısı test olur
        half = max(1, len(split["val"]) // 2)
        split["test"], split["val"] = split["val"][:half], split["val"][half:]

    counts = {}
    for part, xmls in split.items():
        img_dir = os.path.join(args.root, "Insight-MVT_Annotation_Test" if part == "test" and te_xml else "Insight-MVT_Annotation_Train")
        oi, ol = os.path.join(args.out, "images", part), os.path.join(args.out, "labels", part)
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        counts[part] = sum(convert_sequence(x, img_dir, oi, ol, args.every) for x in xmls)
        print(f"{part}: {len(xmls)} dizi, {counts[part]} görüntü")

    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({k: [os.path.basename(x) for x in v] for k, v in split.items()} | {"counts": counts,
                  "every": args.every, "seed": args.seed}, f, indent=2)
    yaml_path = os.path.join(args.out, "ua_detrac.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for k, v in COCO_NAMES.items():
            f.write(f"  {k}: {v}\n")
    print(f"✅ Veri seti yapılandırması: {yaml_path}")


if __name__ == "__main__":
    main()
