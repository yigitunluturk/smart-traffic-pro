"""
Kamera Kayma / Titreşim Telafisi (Camera Motion Compensation)

Sabit olduğu varsayılan kamera rüzgâr, köprü titreşimi veya elde çekim nedeniyle
zamanla kayabilir. Kalibrasyon (homografi) ilk karede yapıldığı için bu kayma
doğrudan hız hatasına dönüşür. Bu modül her karede, araç OLMAYAN arka plan
noktalarını (ORB) referans kareyle eşleştirip "mevcut kare -> referans kare"
dönüşümünü (benzerlik / kısmi afin) hesaplar. Araç konumları bu dönüşümle
referans kareye taşınır ve kalibrasyon her zaman geçerli kalır.
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(
        self,
        enabled: bool = True,
        work_width: int = 640,
        n_features: int = 800,
        min_inliers: int = 20,
        max_shift_ratio: float = 0.25,
        every_n: int = 3,
    ):
        self.enabled = enabled
        self.work_width = work_width
        self.min_inliers = min_inliers
        self.max_shift_ratio = max_shift_ratio
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.scale: float = 1.0
        self.ref_kp = None
        self.ref_des = None
        self.prev_kp = None
        self.prev_des = None
        # Tam çözünürlükte 3x3 dönüşümler
        self.M_cur_to_ref = np.eye(3)
        self.M_prev_to_ref = np.eye(3)
        self.last_inliers = 0
        self.last_ok = True
        # Kamera kayması yavaş bir olaydır: dönüşüm her N karede bir yeniden hesaplanır (CPU tasarrufu)
        self.every_n = max(1, int(every_n))
        self._calls = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.ref_kp = self.ref_des = self.prev_kp = self.prev_des = None
        self.M_cur_to_ref = np.eye(3)
        self.M_prev_to_ref = np.eye(3)

    def _features(self, frame: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]):
        h, w = frame.shape[:2]
        self.scale = self.work_width / float(w) if w > self.work_width else 1.0
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA) \
            if self.scale != 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        mask = np.full(gray.shape, 255, np.uint8)
        # Hareketli araçları maskele (biraz genişleterek)
        for (x1, y1, x2, y2) in boxes:
            pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            cv2.rectangle(
                mask,
                (int((x1 - pw) * self.scale), int((y1 - ph) * self.scale)),
                (int((x2 + pw) * self.scale), int((y2 + ph) * self.scale)),
                0, -1,
            )
        kp, des = self.orb.detectAndCompute(gray, mask)
        return kp, des

    def _estimate(self, kp_a, des_a, kp_b, des_b) -> Tuple[Optional[np.ndarray], int]:
        """a karesindeki noktaları b karesine taşıyan 3x3 matris (küçük ölçekte)."""
        if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return None, 0
        knn = self.matcher.knnMatch(des_a, des_b, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        src = np.float32([kp_a[m.queryIdx].pt for m in good])
        dst = np.float32([kp_b[m.trainIdx].pt for m in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995)
        if A is None or inl is None:
            return None, 0
        n_in = int(inl.sum())
        if n_in < self.min_inliers:
            return None, n_in
        M = np.vstack([A, [0, 0, 1]])
        return M, n_in

    def _to_full(self, M_small: np.ndarray) -> np.ndarray:
        S = np.diag([self.scale, self.scale, 1.0])
        return np.linalg.inv(S) @ M_small @ S

    def update(self, frame: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]] = ()) -> np.ndarray:
        """Kareyi işler ve 'mevcut -> referans' dönüşümünü döndürür."""
        if not self.enabled:
            return self.M_cur_to_ref
        self._calls += 1
        if self.ref_des is not None and (self._calls - 1) % self.every_n:
            return self.M_cur_to_ref
        kp, des = self._features(frame, boxes)

        if self.ref_des is None:
            self.ref_kp, self.ref_des = kp, des
            self.prev_kp, self.prev_des = kp, des
            self.M_cur_to_ref = np.eye(3)
            self.M_prev_to_ref = np.eye(3)
            return self.M_cur_to_ref

        h, w = frame.shape[:2]
        M_small, n_in = self._estimate(kp, des, self.ref_kp, self.ref_des)
        M = None
        if M_small is not None:
            M = self._to_full(M_small)
        else:
            # Referansla doğrudan eşleşme olmadı: bir önceki kare üzerinden zincirle
            M_prev_small, n_in = self._estimate(kp, des, self.prev_kp, self.prev_des)
            if M_prev_small is not None:
                M = self.M_prev_to_ref @ self._to_full(M_prev_small)

        if M is not None:
            # Mantıksız sıçramaları reddet
            shift = np.hypot(M[0, 2], M[1, 2])
            sc = np.sqrt(abs(np.linalg.det(M[:2, :2])))
            if shift > self.max_shift_ratio * max(w, h) or not (0.85 < sc < 1.15):
                M = None

        self.last_ok = M is not None
        self.last_inliers = n_in
        if M is not None:
            self.M_cur_to_ref = M
        # M bulunamazsa son geçerli dönüşüm korunur (kamera kısa sürede sıçramaz)
        self.M_prev_to_ref = self.M_cur_to_ref
        self.prev_kp, self.prev_des = kp, des
        return self.M_cur_to_ref

    def to_reference(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) noktaları referans kare koordinatına taşır."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not self.enabled:
            return pts
        ph = np.hstack([pts, np.ones((len(pts), 1))]) @ self.M_cur_to_ref.T
        return ph[:, :2] / ph[:, 2:3]

    def from_reference(self, pts: np.ndarray) -> np.ndarray:
        """Referans koordinatındaki noktaları mevcut kareye taşır (çizim için)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not self.enabled:
            return pts
        Minv = np.linalg.inv(self.M_cur_to_ref)
        ph = np.hstack([pts, np.ones((len(pts), 1))]) @ Minv.T
        return ph[:, :2] / ph[:, 2:3]
