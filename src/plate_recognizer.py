from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import re

import cv2
import numpy as np


@dataclass
class PlateCandidate:
    text: str
    confidence: float
    char_confs: Optional[List[float]] = None


@dataclass
class PlateTrackState:
    candidates: List[PlateCandidate] = field(default_factory=list)
    locked_text: Optional[str] = None
    locked_conf: Optional[float] = None
    running_text: Optional[str] = None
    last_plate_box: Optional[Tuple[int, int, int, int]] = None  
    last_plate_frame: int = -1                                 
    is_locked: bool = False
    n_attempts: int = 0
    last_frame: int = -1


@dataclass
class PlateDetectionResult:
    plate_text: Optional[str]
    confidence: Optional[float]
    is_locked: bool
    plate_box: Optional[Tuple[int, int, int, int]]
    running_text: Optional[str]


# ----------------------------------------------------------------------
# Türk plakası format kuralları
# ----------------------------------------------------------------------
TR_LETTERS = set("ABCDEFGHIJKLMNOPRSTUVYZ")  # Q, W, X ve Türkçe karakterler kullanılmaz
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "T": "7", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T",
             "8": "B", "Q": "O", "W": "V", "X": "K"}

TR_LAYOUTS = {1: (4,), 2: (3, 4), 3: (2, 3)}
TR_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-PR-VYZ]{1,3})([0-9]{2,4})$")


def is_valid_tr(t: str) -> bool:
    m = TR_REGEX.match(t)
    if not m:
        return False
    letters, digits = m.group(2), m.group(3)
    if any(c not in TR_LETTERS for c in letters):
        return False
    return len(digits) in TR_LAYOUTS[len(letters)]


def correct_tr_plate(text: str, char_confs: Optional[Sequence[float]] = None,
                     max_subs: int = 2) -> Optional[Tuple[str, int]]:

    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if len(t) >= 9 and t.startswith("TR"):
        t = t[2:]
        if char_confs is not None:
            char_confs = list(char_confs)[2:]
    n = len(t)
    if not (7 <= n <= 8):
        return None
    confs = list(char_confs) if char_confs is not None and len(char_confs) == n else [0.5] * n

    best = None  # (cost, subs, text)
    for n_let, digit_opts in TR_LAYOUTS.items():
        for n_dig in digit_opts:
            if 2 + n_let + n_dig != n:
                continue
            out, subs, cost, ok = [], 0, 0.0, True
            for i, c in enumerate(t):
                want_digit = i < 2 or i >= 2 + n_let
                if want_digit:
                    if c.isdigit():
                        out.append(c)
                    elif c in TO_DIGIT:
                        out.append(TO_DIGIT[c]); subs += 1; cost += confs[i]
                    else:
                        ok = False; break
                else:
                    if c in TR_LETTERS:
                        out.append(c)
                    elif c in TO_LETTER:
                        out.append(TO_LETTER[c]); subs += 1; cost += confs[i]
                    else:
                        ok = False; break
            if not ok or subs > max_subs:
                continue
            cand = "".join(out)
            if not is_valid_tr(cand):
                continue
            key = (cost, subs)
            if best is None or key < best[0]:
                best = (key, subs, cand)
    if best is None:
        return None
    return best[2], best[1]


def layout_signature(t: str) -> str:
    return "".join("9" if c.isdigit() else "A" for c in t)


class PlateRecognizer:

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "cct-s-v2-global-model",
        zone_top_ratio: float = 0.40,
        zone_bottom_ratio: float = 0.95,
        min_ocr_conf: float = 0.40,
        n_valid_reads: int = 3,
        plate_format: str = "all",
        sample_every: int = 2,
        min_vehicle_width_ratio: float = 0.03,
        max_reads: int = 15,
        min_agreement: float = 0.5,
        upscale_min_width: int = 360,
        init_alpr: bool = True,
        max_ocr_per_frame: int = 4,
        cpu_only: bool = True,
    ):
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self.zone_top_ratio = zone_top_ratio
        self.zone_bottom_ratio = zone_bottom_ratio
        self.min_ocr_conf = min_ocr_conf
        self.n_valid_reads = n_valid_reads
        self.plate_format = (plate_format or "all").lower()
        self.sample_every = max(1, int(sample_every))
        self.min_vehicle_width_ratio = min_vehicle_width_ratio
        self.max_reads = max_reads
        self.min_agreement = min_agreement
        self.upscale_min_width = upscale_min_width
        self.max_ocr_per_frame = max(1, int(max_ocr_per_frame))
        self.cpu_only = cpu_only
        self._budget_frame = -1
        self._budget_used = 0

        self.alpr = None
        if init_alpr:
            self._init_alpr()

        self.track_states: Dict[int, PlateTrackState] = {}

    def _init_alpr(self):
        try:
            import onnxruntime as ort
            try:
                ort.preload_dlls()
            except Exception:
                pass
            from fast_alpr import ALPR
            if self.cpu_only:
                try:
                    self.alpr = ALPR(detector_model=self.detector_model, ocr_model=self.ocr_model,
                                     detector_providers=["CPUExecutionProvider"],
                                     ocr_providers=["CPUExecutionProvider"])
                    return
                except TypeError:
                    pass
            self.alpr = ALPR(detector_model=self.detector_model, ocr_model=self.ocr_model)
        except Exception as e:
            print(f"[PlateRecognizer] fast-alpr başlatılırken hata / uyarı: {e}")
            self.alpr = None

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_plate(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", (text or "").upper())

    def is_valid_plate(self, text: str) -> bool:
        t = self.normalize_plate(text)
        if not (4 <= len(t) <= 10):
            return False
        if not (any(c.isdigit() for c in t) and any(c.isalpha() for c in t)):
            return False
        if self.plate_format == "tr":
            return is_valid_tr(t)
        return True

    def clean_read(self, raw_text: str, char_confs: Optional[Sequence[float]] = None) -> Optional[str]:
        """Formata göre ham okumayı düzeltir; geçersizse None döner."""
        t = self.normalize_plate(raw_text)
        if self.plate_format in ("tr", "auto"):
            fixed = correct_tr_plate(t, char_confs)
            if fixed is not None:
                return fixed[0]
            if self.plate_format == "tr":
                return None
        return t if self.is_valid_plate(t) else None

    # ------------------------------------------------------------------
    # Oylama
    # ------------------------------------------------------------------
    def _vote(self, cands: List[PlateCandidate]):
        if not cands:
            return None
        score: Dict[str, float] = {}
        count: Dict[str, int] = {}
        for c in cands:
            score[c.text] = score.get(c.text, 0.0) + c.confidence
            count[c.text] = count.get(c.text, 0) + 1
        leader = max(score, key=lambda k: (score[k], count[k]))
        confs = [c.confidence for c in cands if c.text == leader]
        return leader, count[leader], len(cands), float(np.mean(confs))

    @staticmethod
    def _fuse(cands: List[PlateCandidate], leader: str) -> str:
        sig = layout_signature(leader)
        same = [c for c in cands if len(c.text) == len(leader) and layout_signature(c.text) == sig]
        if len(same) < 3:
            return leader
        out = []
        for i in range(len(leader)):
            w: Dict[str, float] = {}
            for c in same:
                cc = c.char_confs[i] if c.char_confs and len(c.char_confs) == len(c.text) else c.confidence
                w[c.text[i]] = w.get(c.text[i], 0.0) + cc
            out.append(max(w, key=w.get))
        return "".join(out)

    def _try_lock(self, st: PlateTrackState, final: bool = False):
        v = self._vote(st.candidates)
        if v is None:
            return
        leader, votes, total, mean_conf = v
        agreement = votes / max(1, total)
        lock = False
        if votes >= self.n_valid_reads and agreement >= self.min_agreement and mean_conf >= self.min_ocr_conf:
            lock = True
        elif (final or total >= self.max_reads) and (
            (votes >= 2 and agreement >= 0.4) or (votes == 1 and total <= 2 and mean_conf >= 0.85)
        ):
            lock = True
        if not lock:
            return
        text = self._fuse(st.candidates, leader)
        if self.plate_format in ("tr",) and not is_valid_tr(text):
            text = leader
        st.locked_text = text
        st.locked_conf = round(mean_conf, 2)
        st.is_locked = True

    def finalize_track(self, track_id: int) -> Optional[str]:
        st = self.track_states.get(track_id)
        if st is None:
            return None
        if not st.is_locked:
            self._try_lock(st, final=True)
        return st.locked_text

    # ------------------------------------------------------------------
    def process_vehicle(
        self,
        track_id: int,
        vehicle_box: Tuple[int, int, int, int],
        vehicle_center: Tuple[int, int],
        frame_clean: np.ndarray,
        frame_height: int,
        frame_no: int = 0
    ) -> PlateDetectionResult:
        if track_id not in self.track_states:
            self.track_states[track_id] = PlateTrackState()

        st = self.track_states[track_id]
        st.last_frame = frame_no
        cx, cy = vehicle_center
        x1, y1, x2, y2 = vehicle_box
        H, W = frame_clean.shape[:2]

        zone_y_top = int(frame_height * self.zone_top_ratio)
        zone_y_bottom = int(frame_height * self.zone_bottom_ratio)
        in_zone = zone_y_top <= cy <= zone_y_bottom

        should_sample = ((frame_no + track_id) % self.sample_every == 0) if frame_no > 0 else True
        w_box, h_box = x2 - x1, y2 - y1
        big_enough = w_box >= max(50, self.min_vehicle_width_ratio * W) and h_box >= 35

        if frame_no != self._budget_frame:
            self._budget_frame, self._budget_used = frame_no, 0
        has_budget = self._budget_used < self.max_ocr_per_frame

        if in_zone and not st.is_locked and self.alpr is not None and should_sample and big_enough and has_budget:
            self._budget_used += 1
            px, py = int(0.04 * w_box), int(0.04 * h_box)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
            if (cx2 - cx1 > 40) and (cy2 - cy1 > 30):
                car_crop = frame_clean[cy1:cy2, cx1:cx2]
                st.n_attempts += 1
                best_read = self._predict_plate(car_crop)
                if best_read is not None:
                    ptext, pconf, (px1, py1, px2, py2), cconfs = best_read
                    st.last_plate_box = (cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2)
                    st.last_plate_frame = frame_no
                    st.running_text = ptext
                    if pconf >= self.min_ocr_conf:
                        st.candidates.append(PlateCandidate(text=ptext, confidence=pconf, char_confs=cconfs))
                        self._try_lock(st)

        if st.locked_text:
            active_text, active_conf = st.locked_text, st.locked_conf
        else:
            v = self._vote(st.candidates)
            active_text = v[0] if v else None
            active_conf = round(v[3], 2) if v else None

        return PlateDetectionResult(
            plate_text=active_text,
            confidence=active_conf,
            is_locked=st.is_locked,
            plate_box=st.last_plate_box if st.last_plate_frame == frame_no else None,
            running_text=st.running_text
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _ocr_confidences(c) -> Tuple[float, Optional[List[float]]]:
        if isinstance(c, (list, tuple, np.ndarray)):
            arr = [float(x) for x in np.asarray(c).ravel()]
            if not arr:
                return 0.0, None
            return float(np.mean(arr)), arr
        return float(c), None

    def _predict_plate(self, car_crop: np.ndarray):
        if car_crop is None or car_crop.size == 0 or self.alpr is None:
            return None
        scale = 1.0
        img = car_crop
        if car_crop.shape[1] < self.upscale_min_width:
            scale = min(2.5, self.upscale_min_width / float(car_crop.shape[1]))
            img = cv2.resize(car_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            results = self.alpr.predict(img)
        except Exception:
            return None

        best = None
        for r in results or []:
            if getattr(r, "ocr", None) is None or not r.ocr.text:
                continue
            conf, cconfs = self._ocr_confidences(r.ocr.confidence)
            raw = self.normalize_plate(r.ocr.text)
            if cconfs is not None and len(cconfs) != len(raw):
                cconfs = None
            clean = self.clean_read(raw, cconfs)
            if clean is None:
                continue
            if cconfs is not None and len(cconfs) != len(clean):
                cconfs = None
            det_conf = float(getattr(r.detection, "confidence", 1.0) or 1.0)
            score = conf * (0.5 + 0.5 * det_conf)
            bb = r.detection.bounding_box
            box = (int(bb.x1 / scale), int(bb.y1 / scale), int(bb.x2 / scale), int(bb.y2 / scale))
            if best is None or score > best[4]:
                best = (clean, conf, box, cconfs, score)
        if best is None:
            return None
        return best[0], best[1], best[2], best[3]
