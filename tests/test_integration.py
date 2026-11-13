import os
import tempfile
import cv2
import numpy as np
import pytest
from src.pipeline import TrafficPipeline


def test_pipeline_end_to_end_integration():

    pipeline = TrafficPipeline("config/default_config.yaml")
    
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (720, 565), (820, 620), (100, 100, 100), -1)

    annotated, summary = pipeline.process_frame(frame, frame_no=1, fps=30.0)

    assert annotated.shape == frame.shape
    assert isinstance(summary, dict)
    assert "total_vehicles" in summary
    assert "violations_count" in summary


def test_pipeline_stream_integration():

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_name = tmp_vid.name

    writer = cv2.VideoWriter(tmp_name, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480))
    for _ in range(10):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(f)
    writer.release()

    pipeline = TrafficPipeline("config/default_config.yaml")
    frames_processed = 0
    for annotated, f_idx, summary in pipeline.stream_video(tmp_name):
        frames_processed += 1
        assert annotated is not None
        assert f_idx == frames_processed

    assert frames_processed == 10

    if os.path.exists(tmp_name):
        os.remove(tmp_name)
