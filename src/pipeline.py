import os
from typing import Callable, Dict, Generator, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

from .detector import VehicleDetector
from .speed_estimator import SpeedEstimator
from .plate_recognizer import PlateRecognizer
from .violation_manager import ViolationManager
from .visualizer import FrameVisualizer


class TrafficPipeline:
    """
    Hız ve Plaka Tespiti Uçtan Uca Entegre Pipeline Sınıfı.
    """
    def __init__(self, config: Union[str, Dict]):
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = config

        # 1. Modül Başlatmaları
        model_cfg = self.cfg.get("model", {})
        self.detector = VehicleDetector(
            model_path=model_cfg.get("vehicle_model", "yolo11m.pt"),
            classes=model_cfg.get("vehicle_classes", [2, 3, 5, 7]),
            conf_threshold=model_cfg.get("vehicle_conf", 0.40),
            tracker=model_cfg.get("tracker", "botsort.yaml"),
            device=model_cfg.get("device", "cpu")
        )

        speed_cfg = self.cfg.get("speed", {})
        self.speed_limit = speed_cfg.get("speed_limit", 80)
        base_res = speed_cfg.get("base_resolution", [1280, 720])
        self.speed_estimator = SpeedEstimator(
            zones_config=speed_cfg.get("zones", []),
            speed_limit=self.speed_limit,
            base_resolution=tuple(base_res) if base_res else None
        )

        plate_cfg = self.cfg.get("plate", {})
        self.plate_recognizer = PlateRecognizer(
            detector_model=plate_cfg.get("detector_model", "yolo-v9-t-384-license-plate-end2end"),
            ocr_model=plate_cfg.get("ocr_model", "cct-s-v2-global-model"),
            zone_top_ratio=plate_cfg.get("zone_top_ratio", 0.40),
            zone_bottom_ratio=plate_cfg.get("zone_bottom_ratio", 0.95),
            min_ocr_conf=plate_cfg.get("min_ocr_conf", 0.40),
            n_valid_reads=plate_cfg.get("n_valid_reads", 3),
            plate_format=plate_cfg.get("plate_format", "all")
        )

        out_cfg = self.cfg.get("output", {})
        self.violation_manager = ViolationManager(
            speed_limit=self.speed_limit,
            save_csv=out_cfg.get("save_csv", True),
            csv_path=out_cfg.get("csv_path", "output/arac_kayitlari.csv"),
            save_snapshots=out_cfg.get("save_violation_snapshots", True),
            violations_dir=out_cfg.get("violations_dir", "output/ihlaller")
        )

        self.visualizer = FrameVisualizer(
            hud_enabled=out_cfg.get("hud_stats", True),
            speed_limit=self.speed_limit
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_no: int,
        fps: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Tek bir video karesini işler: araç tespiti + hız hesaplama + plaka tanıma + çizim.
        """
        clean_frame = frame.copy()
        h, w = frame.shape[:2]

        # Çözünürlüğe göre poligonları uyarla
        self.speed_estimator.adapt_to_frame(w, h)

        # 1. Araç Tespiti ve Takibi
        detections = self.detector.detect_and_track(frame)

        # 2. Hız Bölgelerini ve Plaka Bandını Çiz
        self.visualizer.draw_zones(frame, self.speed_estimator.get_zones())
        self.visualizer.draw_plate_ocr_band(
            frame,
            self.plate_recognizer.zone_top_ratio,
            self.plate_recognizer.zone_bottom_ratio
        )

        # 3. Her Tespit Edilen Araç İçin Analiz
        for det in detections:
            # Hız Güncelleme
            speed_info = self.speed_estimator.update(
                track_id=det.track_id,
                center=det.center,
                frame_no=frame_no,
                fps=fps
            )

            # Plaka Tanıma
            plate_res = self.plate_recognizer.process_vehicle(
                track_id=det.track_id,
                vehicle_box=det.box,
                vehicle_center=det.center,
                frame_clean=clean_frame,
                frame_height=h
            )

            # İhlal & Kayıt Yönetimi
            self.violation_manager.register_or_update(
                track_id=det.track_id,
                class_name=det.class_name,
                frame_no=frame_no,
                speed_kmh=speed_info.speed_kmh,
                lane_name=speed_info.lane_name,
                elapsed_time=speed_info.elapsed_time,
                plate_text=plate_res.plate_text,
                plate_conf=plate_res.confidence,
                frame_for_snapshot=clean_frame,
                vehicle_box=det.box
            )

            # Araç Bilgilerini Ekrana Çiz
            self.visualizer.draw_vehicle(frame, det, speed_info, plate_res)

        # 4. HUD İstatistik Paneli
        summary = self.violation_manager.get_summary()
        self.visualizer.draw_hud(frame, frame_no, fps, summary)

        return frame, summary

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_window: bool = False,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> Dict:
        """
        Tüm videoyu baştan sona işler, opsiyonel olarak çıktı MP4 videosu yazar.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        frame_no = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_no += 1
                annotated, summary = self.process_frame(frame, frame_no, fps)

                if writer:
                    writer.write(annotated)

                if show_window:
                    # Ekrana sığması için gerekiyorsa yeniden boyutlandır
                    display_frame = annotated
                    if width > 1920 or height > 1080:
                        display_frame = cv2.resize(annotated, (1280, 720))
                    cv2.imshow("Trafik Hiz ve Plaka Tespiti", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if progress_callback:
                    progress_callback(frame_no, total_frames, summary)
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

        # Sonuçları CSV'ye kaydet
        csv_file = self.violation_manager.export_csv()
        final_summary = self.violation_manager.get_summary()
        final_summary["csv_path"] = csv_file
        if output_path:
            final_summary["output_video"] = output_path

        return final_summary

    def stream_video(
        self,
        video_source: Union[str, int]
    ) -> Generator[Tuple[np.ndarray, int, Dict], None, None]:
        """
        Canlı yayın veya Streamlit vb. için generator arayüzü.
        """
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_no = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1
                annotated, summary = self.process_frame(frame, frame_no, fps)
                yield annotated, frame_no, summary
        finally:
            cap.release()
