import os
import uuid
import pytest
from src.database import TrafficDatabase


@pytest.fixture
def temp_db():
    db_path = f"output/test_{uuid.uuid4().hex[:8]}.db"
    db = TrafficDatabase(db_path=db_path)
    yield db
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_database_init_and_tables(temp_db):
    assert os.path.exists(temp_db.db_path)
    records = temp_db.get_all_records()
    assert len(records) == 0


def test_session_creation(temp_db):
    session_id = temp_db.create_session("video.mp4", speed_limit=70)
    assert session_id == 1


def test_save_and_update_vehicle(temp_db):
    session_id = temp_db.create_session("test.mp4", 80)
    
    vehicle_data = {
        "track_id": 101,
        "class_name": "car",
        "speed_kmh": 95,
        "speed_limit": 80,
        "is_speeding": True,
        "plate_text": "34ABC123",
        "plate_confidence": 0.92,
        "lane_name": "Right_Lane",
        "elapsed_time_sec": 0.35,
        "first_seen_frame": 10,
        "last_seen_frame": 25,
        "timestamp": "2026-10-05 12:00:00"
    }

    temp_db.save_or_update_vehicle(vehicle_data, session_id=session_id)
    records = temp_db.get_all_records(session_id=session_id)
    assert len(records) == 1
    assert records[0]["track_id"] == 101
    assert records[0]["speed_kmh"] == 95
    assert records[0]["plate_text"] == "34ABC123"
    assert records[0]["is_speeding"] == 1

    # Update the record
    vehicle_data["last_seen_frame"] = 40
    vehicle_data["plate_confidence"] = 0.96
    temp_db.save_or_update_vehicle(vehicle_data, session_id=session_id)

    updated_records = temp_db.get_all_records(session_id=session_id)
    assert len(updated_records) == 1
    assert updated_records[0]["last_seen_frame"] == 40
    assert updated_records[0]["plate_confidence"] == 0.96


def test_search_by_plate(temp_db):
    rec1 = {"track_id": 1, "class_name": "car", "plate_text": "06ANK99", "is_speeding": False}
    rec2 = {"track_id": 2, "class_name": "truck", "plate_text": "34IST01", "is_speeding": True}
    temp_db.save_or_update_vehicle(rec1)
    temp_db.save_or_update_vehicle(rec2)

    found = temp_db.search_by_plate("ANK")
    assert len(found) == 1
    assert found[0]["plate_text"] == "06ANK99"

    violations = temp_db.get_violations()
    assert len(violations) == 1
    assert violations[0]["track_id"] == 2


def test_update_plate_text(temp_db):
    rec = {"track_id": 5, "class_name": "car", "plate_text": "34WRONG", "is_speeding": False}
    temp_db.save_or_update_vehicle(rec)
    records = temp_db.get_all_records()
    record_id = records[0]["id"]

    success = temp_db.update_plate_text(record_id, "34RIGHT123")
    assert success is True

    updated_records = temp_db.get_all_records()
    assert updated_records[0]["plate_text"] == "34RIGHT123"


def test_watchlist_operations(temp_db):
    # Varsayılan demo kayıtları kontrol et
    watchlist = temp_db.get_watchlist()
    assert len(watchlist) >= 2

    # Yeni aranan araç ekleme
    success = temp_db.add_to_watchlist("06TEST99", "Çalıntı İhbarı", "danger")
    assert success is True

    # Eşleşme kontrolü (boşluksuz ve küçük/büyük harf duyarsız)
    match = temp_db.check_watchlist("06 test 99")
    assert match is not None
    assert match["plate_text"] == "06TEST99"
    assert match["reason"] == "Çalıntı İhbarı"

    # Silme işlemi
    item_id = match["id"]
    removed = temp_db.remove_from_watchlist(item_id)
    assert removed is True

    # Silindikten sonra eşleşmemeli
    assert temp_db.check_watchlist("06TEST99") is None


def test_analytics_stats(temp_db):
    temp_db.save_or_update_vehicle({"track_id": 1, "class_name": "car", "speed_kmh": 60, "is_speeding": False})
    temp_db.save_or_update_vehicle({"track_id": 2, "class_name": "truck", "speed_kmh": 90, "is_speeding": True})
    temp_db.save_or_update_vehicle({"track_id": 3, "class_name": "motorcycle", "speed_kmh": 110, "is_speeding": True})

    stats = temp_db.get_analytics_stats()
    assert stats["total_vehicles"] == 3
    assert stats["total_violations"] == 2
    assert stats["violation_rate"] == 66.7
    assert stats["max_speed"] == 110
    assert stats["speed_distribution"]["81-100 km/s"] == 1
    assert stats["speed_distribution"]["100+ km/s"] == 1
    assert stats["class_distribution"]["Otomobil"] == 1
    assert stats["class_distribution"]["Motosiklet"] == 1
    assert stats["class_distribution"]["Kamyon"] == 1
