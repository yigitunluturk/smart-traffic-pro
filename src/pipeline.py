import json
import os
from typing import Callable, Dict, Generator, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

from .camera_motion import CameraMotionCompensator
from .privacy import PlatePrivacy
from .detector import VehicleDetector
from .speed_estimator import SpeedEstimator
from .plate_recognizer import PlateRecognizer
from .violation_manager import ViolationManager
from .visualizer import FrameVisualizer


class TrafficPipeline:
    """
    Hız ve Plaka Tespiti Uçtan Uca Entegre Pipeline Sınıfı.

    Akış (her kare):
      1. YOLO11 + ByteTrack ile araç tespiti/takibi
      2. Kamera kayma telafisi (arka plan eşleştirme)
      3. Yol düzlemi kalibrasyonu (homografi) ile metre cinsinden konum -> hız
      4. Plaka okuma + oylama
      5. Kadrajdan çıkan araçların hız/plaka/ihlal kaydının kesinleştirilmesi
    """
    def __init__(self, config: Union[str, Dict], video_path: Optional[str] = None):
        if isinstance(config, str):
            self.config_dir = os.path.dirname(os.path.abspath(config))
            with open(config, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.config_dir = os.path.abspath("config")
            self.cfg = config

        # 1. Modül Başlatmaları
        model_cfg = self.cfg.get("model", {})
        self.detector = VehicleDetector(
            model_path=model_cfg.get("vehicle_model", "yolo11m.pt"),
            classes=model_cfg.get("vehicle_classes", [2, 3, 5, 7]),
            conf_threshold=model_cfg.get("vehicle_conf", 0.40),
            tracker=model_cfg.get("tracker", "botsort.yaml"),
            device=model_cfg.get("device", "cpu"),
            imgsz=model_cfg.get("imgsz"),
        )

        speed_cfg = self.cfg.get("speed", {})
        self.speed_cfg = speed_cfg
        self.speed_limit = speed_cfg.get("speed_limit", 80)
        base_res = speed_cfg.get("base_resolution", [1280, 720])
        plate_cfg = self.cfg.get("plate", {})
        top_ratio = float(plate_cfg.get("zone_top_ratio", 0.40))
        bottom_ratio = float(plate_cfg.get("zone_bottom_ratio", 0.95))
        meas = speed_cfg.get("measurement", {})

        self.speed_method = str(speed_cfg.get("method", "auto")).lower()
        self.speed_estimator = SpeedEstimator(
            zones_config=speed_cfg.get("zones", []) if self.speed_method == "zones" else [],
            speed_limit=self.speed_limit,
            base_resolution=tuple(base_res) if base_res else None,
            band_top_ratio=top_ratio,
            band_bottom_ratio=bottom_ratio,
            corridor_real_distance_m=float(speed_cfg.get("corridor_real_distance_m", 24.0)),
            use_corridor_mode=self.speed_method in ("auto", "corridor", "homography"),
            homography_config=speed_cfg.get("homography") if self.speed_method == "homography" else None,
            min_track_time=float(meas.get("min_track_time_s", 0.6)),
            min_track_distance_m=float(meas.get("min_track_distance_m", 5.0)),
            lock_time=float(meas.get("lock_time_s", 1.2)),
            lock_distance_m=float(meas.get("lock_distance_m", 25.0)),
        )
        self.calibration_source: str = "config" if self.speed_estimator.H is not None else "yok (yaklaşık koridor modu)"
        if video_path:
            self.load_calibration_for_video(video_path)

        stab_cfg = speed_cfg.get("stabilization", {})
        self.motion = CameraMotionCompensator(
            enabled=bool(stab_cfg.get("enabled", True)),
            work_width=int(stab_cfg.get("work_width", 640)),
            every_n=int(stab_cfg.get("every_n_frames", 3)),
        )

        self.plate_recognizer = PlateRecognizer(
            detector_model=plate_cfg.get("detector_model", "yolo-v9-t-384-license-plate-end2end"),
            ocr_model=plate_cfg.get("ocr_model", "cct-s-v2-global-model"),
            zone_top_ratio=top_ratio,
            zone_bottom_ratio=bottom_ratio,
            min_ocr_conf=plate_cfg.get("min_ocr_conf", 0.40),
            n_valid_reads=plate_cfg.get("n_valid_reads", 3),
            plate_format=plate_cfg.get("plate_format", "all"),
            sample_every=plate_cfg.get("sample_every", 2),
            min_vehicle_width_ratio=plate_cfg.get("min_vehicle_width_ratio", 0.03),
            max_reads=plate_cfg.get("max_reads", 15),
            max_ocr_per_frame=plate_cfg.get("max_ocr_per_frame", 4),
            # Plaka modelleri (ONNX) hafif: GPU kütüphanesi uyumsuzluklarına karşı her zaman CPU'da çalışır
            cpu_only=str(model_cfg.get("device", "cpu")).lower() in ("cpu", "auto"),
        )

        out_cfg = self.cfg.get("output", {})
        self.violation_manager = ViolationManager(
            speed_limit=self.speed_limit,
            save_csv=out_cfg.get("save_csv", True),
            csv_path=out_cfg.get("csv_path", "output/arac_kayitlari.csv"),
            save_snapshots=out_cfg.get("save_violation_snapshots", True),
            violations_dir=out_cfg.get("violations_dir", "output/ihlaller"),
            db_path=out_cfg.get("database_path", "output/traffic_surveillance.db"),
        )

        self.visualizer = FrameVisualizer(
            hud_enabled=out_cfg.get("hud_stats", True),
            speed_limit=self.speed_limit
        )

        priv_cfg = self.cfg.get("privacy", {})
        self.privacy = PlatePrivacy(enabled=bool(priv_cfg.get("blur_non_violators", True)))
        self.retention_days = int(priv_cfg.get("retention_days", 30))
        try:
            n = self.violation_manager.db.apply_retention(self.retention_days)
            if n:
                print(f"🔒 [Pipeline] Saklama süresi ({self.retention_days} gün) dolan {n} kaydın plakası silindi.")
        except Exception as e:
            print(f"[Pipeline] Saklama politikası uyarısı: {e}")

        # NFR-3: Ayarlanabilir kare atlama (1 = her kare işlenir)
        self.frame_stride = max(1, int(model_cfg.get("frame_stride", 1)))
        self._last_overlays = []

        # İz yaşam döngüsü
        self.last_seen: Dict[int, int] = {}
        self.track_class: Dict[int, str] = {}
        self.lost_after_s = float(speed_cfg.get("track_lost_after_s", 0.7))

    # ------------------------------------------------------------------
    # Kalibrasyon yönetimi
    # ------------------------------------------------------------------
    def calibration_dir(self) -> str:
        d = self.speed_cfg.get("calibration_dir", "calibrations")
        return d if os.path.isabs(d) else os.path.join(self.config_dir, d)

    def load_calibration_for_video(self, video_path: Union[str, int]) -> bool:
        """
        config/calibrations/<video_adı>.json varsa o videoya özel yol düzlemi
        kalibrasyonunu yükler. Kalibrasyon kameraya/sahneye özeldir.
        """
        if not isinstance(video_path, str):
            return False
        stem = os.path.splitext(os.path.basename(video_path))[0]
        path = os.path.join(self.calibration_dir(), f"{stem}.json")
        if not os.path.exists(path):
            if self.speed_estimator.H is None:
                print(f"⚠️ [Pipeline] '{stem}' için kalibrasyon yok -> yaklaşık koridor modu. "
                      f"Doğru hız için: python zone_calibrator.py --input \"{video_path}\"")
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            self.speed_estimator.set_homography(cal)
            self.calibration_source = os.path.relpath(path)
            print(f"✅ [Pipeline] Yol düzlemi kalibrasyonu yüklendi: {path}")
            return True
        except Exception as e:
            print(f"⚠️ [Pipeline] Kalibrasyon okunamadı ({path}): {e}")
            return False

    # ------------------------------------------------------------------
    def _finalize_track(self, tid: int):
        info = self.speed_estimator.finalize_track(tid)
        plate = self.plate_recognizer.finalize_track(tid)
        st = self.plate_recognizer.track_states.get(tid)
        self.violation_manager.finalize_track(
            tid,
            plate_text=plate,
            plate_conf=st.locked_conf if st else None,
            speed_info=info,
        )
        self.last_seen.pop(tid, None)
        self.visualizer.trails.pop(tid, None)
        self.privacy.forget(tid)

    def _finalize_lost_tracks(self, frame_no: int, fps: float):
        limit = max(5, int(self.lost_after_s * (fps or 30.0)))
        for tid, last in list(self.last_seen.items()):
            if frame_no - last > limit:
                self._finalize_track(tid)

    def finalize_all(self):
        for tid in list(self.last_seen.keys()):
            self._finalize_track(tid)
        self.violation_manager.flush()

    def _draw_calibration(self, frame: np.ndarray):
        est = self.speed_estimator
        if est.H is None:
            return
        h, w = frame.shape[:2]
        try:
            img_pts = np.array(est.homography_cfg["image_points"], dtype=np.float64)
            bw, bh = est.calib_base_res
            pts = img_pts * [w / bw, h / bh]
            pts = self.motion.from_reference(pts)
            cv2.polylines(frame, [pts.astype(np.int32).reshape(-1, 1, 2)], True, (255, 200, 0), max(1, h // 540), cv2.LINE_AA)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        frame_no: int,
        fps: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Tek bir video karesini işler: araç tespiti + hız hesaplama + plaka tanıma + çizim.
        """
        h, w = frame.shape[:2]

        # Atlanan karelerde analiz yapılmaz; son bilinen kutular/etiketler çizilir.
        # Hız zaman damgası (frame_no / fps) ile hesaplandığı için doğruluk etkilenmez.
        if self.frame_stride > 1 and frame_no > 1 and (frame_no - 1) % self.frame_stride != 0:
            self.visualizer.draw_plate_ocr_band(frame, self.plate_recognizer.zone_top_ratio,
                                                self.plate_recognizer.zone_bottom_ratio)
            self._draw_calibration(frame)
            for det, sp, pl in self._last_overlays:
                if self.privacy.enabled and not sp.is_speeding:
                    self.privacy.blur(frame, self.privacy.region_for(det.track_id, det.box, frame.shape))
                self.visualizer.draw_vehicle(frame, det, sp, pl)
            summary = self.violation_manager.get_summary()
            self.visualizer.draw_hud(frame, frame_no, fps, summary)
            return frame, summary

        clean_frame = frame.copy()
        overlays = []

        self.speed_estimator.adapt_to_frame(w, h)

        # 1. Araç Tespiti ve Takibi
        detections = self.detector.detect_and_track(frame)

        # 2. Kamera kayma telafisi (yalnızca kalibrasyon varsa gerekli)
        if self.speed_estimator.H is not None:
            self.motion.update(clean_frame, [d.box for d in detections])

        # 3. Bölge çizimleri
        self.visualizer.draw_plate_ocr_band(
            frame,
            self.plate_recognizer.zone_top_ratio,
            self.plate_recognizer.zone_bottom_ratio
        )
        self._draw_calibration(frame)

        # 4. Analiz (hız + plaka) — çizimden önce, çünkü gizlilik kararı ihlal durumuna bağlı
        results = []
        for det in detections:
            self.last_seen[det.track_id] = frame_no
            x1, y1, x2, y2 = det.box
            ground_ref = None
            if self.speed_estimator.H is not None:
                ground_ref = tuple(self.motion.to_reference(np.array([[(x1 + x2) / 2.0, float(y2)]]))[0])

            speed_info = self.speed_estimator.update(
                track_id=det.track_id,
                center=det.center,
                frame_no=frame_no,
                fps=fps,
                ground_point_ref=ground_ref,
                box=det.box,
            )
            plate_res = self.plate_recognizer.process_vehicle(
                track_id=det.track_id,
                vehicle_box=det.box,
                vehicle_center=det.center,
                frame_clean=clean_frame,
                frame_height=h,
                frame_no=frame_no
            )
            self.privacy.observe_plate(det.track_id, det.box, plate_res.plate_box)
            results.append((det, speed_info, plate_res))

        # 5. KVKK (FR-07): ihlal yapmayan araçların plakaları gösterilen ve kaydedilen
        #    karelerde bulanıklaştırılır; ihlal yapanın plakası kanıt olarak açık kalır.
        evidence_frame = clean_frame
        if self.privacy.enabled and results:
            evidence_frame = clean_frame.copy()
            for det, sp, _ in results:
                if not sp.is_speeding:
                    region = self.privacy.region_for(det.track_id, det.box, frame.shape)
                    self.privacy.blur(evidence_frame, region)
                    self.privacy.blur(frame, region)

        # 6. Kayıt + çizim
        for det, speed_info, plate_res in results:
            # Kayıtlara yalnızca kesinleşmiş (kilitli) plaka "plaka" olarak yazılır;
            # kesinleşmemiş okuma "plate_candidate" olarak düşük güvenle tutulur.
            self.violation_manager.register_or_update(
                track_id=det.track_id,
                class_name=det.class_name,
                frame_no=frame_no,
                speed_kmh=speed_info.speed_kmh,
                lane_name=speed_info.lane_name,
                elapsed_time=speed_info.elapsed_time,
                plate_text=plate_res.plate_text if plate_res.is_locked else None,
                plate_conf=plate_res.confidence if plate_res.is_locked else None,
                frame_for_snapshot=evidence_frame,
                vehicle_box=det.box,
                plate_final=plate_res.is_locked,
                plate_candidate=plate_res.plate_text,
                plate_rel_box=self.privacy.rel_boxes.get(det.track_id),
            )
            self.visualizer.draw_vehicle(frame, det, speed_info, plate_res)
            overlays.append((det, speed_info, plate_res))

        self._last_overlays = overlays

        # 7. Kadrajdan çıkan araçları kesinleştir
        self._finalize_lost_tracks(frame_no, fps)

        # 8. HUD
        summary = self.violation_manager.get_summary()
        self.visualizer.draw_hud(frame, frame_no, fps, summary)

        return frame, summary

    def process_video(
        self,
        video_path: Union[str, int],
        output_path: Optional[str] = None,
        show_window: bool = False,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> Dict:
        """
        Tüm videoyu baştan sona işler, opsiyonel olarak çıktı MP4 videosu yazar.
        """
        if isinstance(video_path, str) and not os.path.exists(video_path):
            raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

        self.load_calibration_for_video(video_path)
        if self.violation_manager.current_session_id is None:
            self.violation_manager.start_session(str(video_path))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")
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
        summary: Dict = {}
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
            self.finalize_all()

        csv_file = self.violation_manager.export_csv()
        final_summary = self.violation_manager.get_summary()
        final_summary["csv_path"] = csv_file
        final_summary["calibration"] = self.calibration_source
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
        self.load_calibration_for_video(video_source)
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
            self.finalize_all()
