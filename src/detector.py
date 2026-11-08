from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from ultralytics import YOLO


@dataclass
class VehicleDetection:
    track_id: int
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]         # (cx, cy)
    width: int
    height: int
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """
    YOLO ve BoT-SORT / ByteTrack tabanlı araç takip ve tespit modülü.
    """
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        classes: Optional[List[int]] = None,
        conf_threshold: float = 0.40,
        tracker: str = "botsort.yaml",
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.classes = classes if classes is not None else [2, 3, 5, 7]
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.device = device
        self.model = YOLO(model_path)

        import torch
        if self.device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 [VehicleDetector] GPU Aktif: {gpu_name} (CUDA:0)")
                try:
                    self.model.to("cuda")
                except Exception as e:
                    print(f"[VehicleDetector] Model to cuda uyarisi: {e}")
            else:
                print("⚠️ [VehicleDetector] CUDA bulunamadı, CPU kullanılıyor.")
                self.device = "cpu"

    def detect_and_track(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Gelen video karesinde araçları tespit eder ve ID'leri takip eder.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            classes=self.classes,
            conf=self.conf_threshold,
            tracker=self.tracker,
            device=self.device,
            verbose=False
        )[0]

        detections: List[VehicleDetection] = []
        if results.boxes is None or results.boxes.id is None:
            return detections

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(boxes_xyxy, track_ids, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cls_name = self.model.names.get(cls_id, f"vehicle_{cls_id}")

            detections.append(
                VehicleDetection(
                    track_id=int(tid),
                    box=(x1, y1, x2, y2),
                    center=(cx, cy),
                    width=w,
                    height=h,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=cls_name
                )
            )

        return detections
