import os
import pytest
from src.report_generator import generate_violation_pdf


def test_generate_violation_pdf(tmp_path):
    output_pdf = str(tmp_path / "test_violation_report.pdf")
    record = {
        "id": 42,
        "track_id": 42,
        "plate_text": "34ABC999",
        "speed_kmh": 105,
        "speed_limit": 80,
        "is_speeding": True,
        "class_name": "car",
        "lane_name": "Sol Şerit",
        "timestamp": "2026-10-22 19:30:00",
        "snapshot_path": None
    }

    pdf_path = generate_violation_pdf(record, output_pdf)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000  # Generated PDF has valid content
