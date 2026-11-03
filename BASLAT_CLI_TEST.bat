@echo off
chcp 65001 > nul
title Trafik Hız ve Plaka Tespiti - CLI Canlı Test
echo ==============================================================================
echo 🚗 Trafik Hız ve Plaka Tespiti Canlı Video İşleme Başlatılıyor...
echo ==============================================================================
echo.

if exist "C:\Users\msi-nb\Desktop\part1.mp4" (
    python main.py --input "C:\Users\msi-nb\Desktop\part1.mp4" --output output/sonuc_part1.mp4 --show
) else if exist "video.mp4" (
    python main.py --input video.mp4 --output output/sonuc.mp4 --show
) else (
    echo [BILGI] Lutfen islenecek video yolunu giriniz veya surukleyip birakiniz:
    set /p VIDEO_YOLU=Video Dosyasi: 
    python main.py --input %VIDEO_YOLU% --output output/sonuc.mp4 --show
)

pause
