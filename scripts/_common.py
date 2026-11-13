"""Evaluation scripts için ortak yardımcılar (sonuç dosyası, sistem bilgisi)."""
import json
import os
import platform
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_DIR = os.path.join(ROOT, "output", "eval")


def system_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # noqa: F401
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False
    return info


def write_result(name: str, metrics: dict, params: dict, per_item=None, notes: str = "") -> str:
    """Sonucu output/eval/<name>.json olarak yazar (Streamlit S5 ekranı bunu gösterir)."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    out = {
        "name": name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "params": params,
        "system": system_info(),
        "notes": notes,
    }
    if per_item is not None:
        out["per_item"] = per_item
    path = os.path.join(EVAL_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Sonuç yazıldı: {path}")
    for k, v in metrics.items():
        print(f"   {k:28s}: {v}")
    return path


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
