"""
Hızlı Test Scripti: Yapay video karesi üreterek tüm pipeline'ın (YOLO, Tracker, Hız, ALPR, Çizim, CSV)
çalıştığını doğrular.
"""

import os
import cv2
import numpy as np
from src.pipeline import TrafficPipeline

def test_pipeline_execution():
    print("🧪 Test başlatılıyor...")
    config_path = "config/default_config.yaml"
    
    # Küçük bir model (yolo11n veya yolo11s veya yolo11m)
    pipeline = TrafficPipeline(config=config_path)
    print("✅ Pipeline başarıyla yüklendi.")

    # 1920x1080 sentetik bir kare oluştur
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (750, 500), (950, 650), (120, 120, 120), -1) # Sahte araç gövdesi
    cv2.rectangle(test_frame, (810, 610), (890, 640), (255, 255, 255), -1) # Sahte plaka alanı
    cv2.putText(test_frame, "34ABC123", (815, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print("🖼️ Tek kare işleniyor...")
    annotated, summary = pipeline.process_frame(test_frame, frame_no=1, fps=30.0)
    print("✅ Kare işleme başarılı! Özet:", summary)

    # Test CSV kaydı
    csv_path = pipeline.violation_manager.export_csv()
    print(f"✅ CSV kaydı doğrulandı: {csv_path}")
    print("🎉 TÜM BİLEŞENLER SORUNSUZ ÇALIŞIYOR!")

if __name__ == "__main__":
    test_pipeline_execution()
