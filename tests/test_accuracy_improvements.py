"""
Doğruluk iyileştirmeleri için birim testleri:
  - Homografi tabanlı hız tahmini (gürültü + perspektif altında)
  - Kamera kayma telafisi
  - Türk plaka doğrulama/düzeltme (RSA Bölüm 5.2)
  - Çoklu okuma oylaması ve karakter bazlı birleştirme
"""
import numpy as np
import cv2
import pytest

from src.speed_estimator import SpeedEstimator, theil_sen_slope
from src.camera_motion import CameraMotionCompensator
from src.plate_recognizer import (PlateRecognizer, PlateCandidate, PlateTrackState,
                                  correct_tr_plate, is_valid_tr)

CALIB = {
    "base_resolution": [1280, 720],
    "image_points": [[467.4, 613.0], [635.4, 613.0], [659.5, 388.0], [586.9, 388.0]],
    "world_points": [[0.0, 0.0], [3.5, 0.0], [3.5, 24.0], [0.0, 24.0]],
    "measure_y_range": [0.3, 0.99],
}


def _simulate(est, true_kmh, fps=30.0, noise_px=1.5, frame_size=(2560, 1440), seed=0, lane_x=1.75):
    rng = np.random.default_rng(seed)
    v = true_kmh / 3.6
    info = None
    for k in range(1, 200):
        Y = -8.0 + v * k / fps                     # kameradan uzaklaşan araç
        if Y > 40:
            break
        p = est.world_to_image(np.array([[lane_x, Y]]), frame_size)[0]
        p = p + rng.normal(0, noise_px * frame_size[1] / 720.0, 2)
        w = 120
        box = (int(p[0] - w / 2), int(p[1] - 80), int(p[0] + w / 2), int(p[1]))
        info = est.update(1, (int(p[0]), int(p[1] - 40)), k, fps, ground_point_ref=tuple(p), box=box)
        if info.is_calculated:
            break
    return info


@pytest.mark.parametrize("true_kmh", [30, 50, 72, 95, 130])
def test_homography_speed_accuracy(true_kmh):
    errs = []
    for seed in range(5):
        est = SpeedEstimator(homography_config=CALIB, speed_limit=80)
        est.adapt_to_frame(2560, 1440)
        info = _simulate(est, true_kmh, seed=seed)
        if not info.is_calculated:
            est.finalize_track(1)
        assert info.is_calculated, "hız kesinleşmeliydi"
        errs.append(abs(info.speed_kmh - true_kmh))
    # NFR-1 hedefi: MAE <= 2 km/h
    assert np.mean(errs) <= 2.0, errs


def test_homography_roundtrip():
    est = SpeedEstimator(homography_config=CALIB)
    w = est.image_to_world(np.array([[635.4 * 2, 613.0 * 2]]), (2560, 1440))[0]
    assert abs(w[0] - 3.5) < 1e-6 and abs(w[1]) < 1e-6


def test_theil_sen_resists_outliers():
    t = np.arange(30) / 30.0
    y = 20.0 * t
    y[5] += 15
    y[17] -= 12
    assert abs(theil_sen_slope(t, y) - 20.0) < 0.5


def test_camera_motion_compensation():
    rng = np.random.default_rng(1)
    base = (rng.random((720, 1280)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    comp = CameraMotionCompensator(enabled=True, work_width=960, every_n=1)
    comp.update(base, [])
    M = np.float32([[1, 0, 25], [0, 1, -12]])          # kamera kaydı
    moved = cv2.warpAffine(base, M, (1280, 720), borderMode=cv2.BORDER_REFLECT)
    comp.update(moved, [])
    p_ref = comp.to_reference(np.array([[625.0, 388.0]]))[0]
    assert abs(p_ref[0] - 600.0) < 1.5 and abs(p_ref[1] - 400.0) < 1.5


@pytest.mark.parametrize("raw,expected", [
    ("34ABC123", "34ABC123"),
    ("34A8C123", "34ABC123"),     # harf bloğunda rakam -> düzeltilir
    ("O6AB1234", "06AB1234"),     # il kodunda O -> 0
    ("34AB12E4", None),           # düzeltilemez
    ("82ABC123", None),           # il kodu > 81
    ("TR34ABC123", "34ABC123"),   # plakadaki mavi bant 'TR' yazısı
])
def test_tr_correction(raw, expected):
    res = correct_tr_plate(raw)
    assert (res[0] if res else None) == expected


def test_tr_regex_rules():
    assert is_valid_tr("34ABC123")
    assert is_valid_tr("06A1234")
    assert not is_valid_tr("34QBC123")   # Q kullanılmaz
    assert not is_valid_tr("34A8C123")   # harf bloğunda rakam
    assert not is_valid_tr("00ABC12")    # il kodu 00


def test_voting_fuses_single_char_errors():
    rec = PlateRecognizer(n_valid_reads=3, plate_format="tr", init_alpr=False)
    st = PlateTrackState()
    # Her okumada farklı tek karakter hatası; doğru metin hiçbir zaman 3 oy almıyor
    reads = ["34ABC128", "34ABC123", "34ABE123", "34ADC123", "34ABC123"]
    for r in reads:
        st.candidates.append(PlateCandidate(r, 0.8))
    rec._try_lock(st, final=True)
    assert st.is_locked and st.locked_text == "34ABC123"


def test_no_lock_on_single_low_conf_read():
    rec = PlateRecognizer(n_valid_reads=3, plate_format="tr", init_alpr=False)
    st = PlateTrackState()
    st.candidates.append(PlateCandidate("34ABC123", 0.55))
    rec._try_lock(st, final=True)
    assert not st.is_locked   # yanlış plaka yazmaktansa "okunamadı"
