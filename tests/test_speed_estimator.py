import numpy as np
import pytest
from src.speed_estimator import SpeedEstimator


@pytest.fixture
def sample_zones():
    return [
        {
            "name": "Test_Lane_Upward",
            "direction": "upward",
            "entry_polygon": [[100, 500], [300, 500], [300, 480], [100, 480]],
            "exit_polygon": [[100, 300], [300, 300], [300, 280], [100, 280]],
            "calib_pixel_length": 100.0,
            "real_distance_m": 10.0,
            "color_entry": [0, 255, 255],
            "color_exit": [0, 0, 255]
        }
    ]


def test_speed_estimator_initialization(sample_zones):
    estimator = SpeedEstimator(zones_config=sample_zones, speed_limit=70)
    assert len(estimator.zones) == 1
    assert estimator.zones[0].name == "Test_Lane_Upward"
    assert estimator.speed_limit == 70


def test_speed_estimator_calculation(sample_zones):
    estimator = SpeedEstimator(zones_config=sample_zones, speed_limit=80)
    
    info1 = estimator.update(track_id=1, center=(200, 490), frame_no=10, fps=30.0)
    assert not info1.is_calculated

    info2 = estimator.update(track_id=1, center=(200, 290), frame_no=25, fps=30.0)
    assert info2.is_calculated
    assert info2.speed_kmh is not None
    assert info2.speed_kmh == 144
    assert info2.is_speeding is True


def test_speed_estimator_resolution_scaling(sample_zones):
    estimator = SpeedEstimator(
        zones_config=sample_zones,
        speed_limit=80,
        base_resolution=(1280, 720)
    )
    estimator.adapt_to_frame(3840, 2160)
    scaled_zone = estimator.zones[0]
    assert scaled_zone.calib_pixel_length == 300.0
