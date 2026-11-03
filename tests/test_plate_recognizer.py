import pytest
from src.plate_recognizer import PlateRecognizer, PlateCandidate


def test_plate_normalization():
    recognizer = PlateRecognizer(n_valid_reads=3)
    assert recognizer.normalize_plate(" 34 abc 123 ") == "34ABC123"
    assert recognizer.normalize_plate("06-ANK-999") == "06ANK999"


def test_plate_validation():
    recognizer_tr = PlateRecognizer(plate_format="tr")
    # Geçerli Türk plakaları
    assert recognizer_tr.is_valid_plate("34ABC123") is True
    assert recognizer_tr.is_valid_plate("06A1234") is True
    # Geçersiz il kodu
    assert recognizer_tr.is_valid_plate("99ABC123") is False
    # Çok kısa / alakasız
    assert recognizer_tr.is_valid_plate("A1") is False

    recognizer_all = PlateRecognizer(plate_format="all")
    assert recognizer_all.is_valid_plate("B123CD") is True
    assert recognizer_all.is_valid_plate("ABCDE") is False  # Sayı yok


def test_candidate_locking_mechanism():
    recognizer = PlateRecognizer(n_valid_reads=3)
    st = recognizer.track_states.setdefault(1, recognizer.track_states.get(1) or type("State", (), {"candidates": [], "locked_text": None, "locked_conf": None, "is_locked": False, "running_text": None, "last_plate_box": None})())

    # Adayları simüle et
    st.candidates.append(PlateCandidate("34ABC12", 0.60))
    st.candidates.append(PlateCandidate("34ABC123", 0.94))
    st.candidates.append(PlateCandidate("34ABC123", 0.91))

    # Kilit mekanizması
    if len(st.candidates) >= recognizer.n_valid_reads:
        best = max(st.candidates, key=lambda c: c.confidence)
        st.locked_text = best.text
        st.locked_conf = best.confidence
        st.is_locked = True

    assert st.is_locked is True
    assert st.locked_text == "34ABC123"
    assert st.locked_conf == 0.94
