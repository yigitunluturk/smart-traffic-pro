@echo off
chcp 65001 > nul
title Şerit ve Hız Bölgesi Kalibrasyon Aracı
echo ==============================================================================
echo 📐 Şerit ve Hız Bölgesi Kalibrasyon Aracı
echo ==============================================================================
echo.
if exist "C:\Users\msi-nb\Desktop\part1.mp4" (
    python zone_calibrator.py --input "C:\Users\msi-nb\Desktop\part1.mp4"
) else (
    set /p VIDEO_YOLU=Kalibre Edilecek Video Dosyasi: 
    python zone_calibrator.py --input %VIDEO_YOLU%
)
pause
