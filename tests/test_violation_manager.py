import os
import shutil
import uuid
import numpy as np
import pytest
from src.violation_manager import ViolationManager


@pytest.fixture
def temp_manager():
    uid = uuid.uuid4().hex[:8]
    csv_p = f"output/test_{uid}.csv"
    viol_d = f"output/test_viol_{uid}"
    db_p = f"output/test_{uid}.db"
    
    vm = ViolationManager(
        speed_limit=80,
        save_csv=True,
        csv_path=csv_p,
        save_snapshots=True,
        violations_dir=viol_d,
        db_path=db_p
    )
    yield vm

    # Cleanup
    for path in [csv_p, db_p]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    if os.path.exists(viol_d):
        try:
            shutil.rmtree(viol_d)
        except OSError:
            pass


def test_violation_registration_and_snapshot(temp_manager):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = (100, 100, 300, 300)

    # Normal araç (hız limiti aşılmadı)
    rec1 = temp_manager.register_or_update(
        track_id=1,
        class_name="car",
        frame_no=10,
        speed_kmh=75,
        lane_name="Right_Lane"
    )
    assert rec1.is_speeding is False
    assert rec1.snapshot_path is None

    # İhlal yapan araç (105 km/h > 80 km/h)
    rec2 = temp_manager.register_or_update(
        track_id=2,
        class_name="car",
        frame_no=15,
        speed_kmh=105,
        plate_text="34XYZ99",
        plate_conf=0.88,
        frame_for_snapshot=frame,
        vehicle_box=box
    )
    assert rec2.is_speeding is True
    assert rec2.snapshot_path is not None
    assert os.path.exists(rec2.snapshot_path)


def test_export_csv_and_summary(temp_manager):
    temp_manager.register_or_update(track_id=1, class_name="car", frame_no=1, speed_kmh=70)
    temp_manager.register_or_update(track_id=2, class_name="truck", frame_no=2, speed_kmh=90, plate_text="06ABC12")

    csv_path = temp_manager.export_csv()
    assert os.path.exists(csv_path)

    summary = temp_manager.get_summary()
    assert summary["total_vehicles"] == 2
    assert summary["speed_measured_count"] == 2
    assert summary["plate_recognized_count"] == 1
    assert summary["violations_count"] == 1
    assert summary["max_speed"] == 90
    assert summary["average_speed"] == 80.0
