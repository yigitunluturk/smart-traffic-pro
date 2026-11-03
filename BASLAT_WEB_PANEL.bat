@echo off
chcp 65001 > nul
title ArgusVision | AI Trafik Komuta Merkezi (Web SOC)
echo ==============================================================================
echo ⚡ ArgusVision AI Trafik Komuta Merkezi Web Sunucusu Başlatılıyor...
echo 🌐 Web Arayüzü: http://localhost:5000
echo ==============================================================================
echo.
start "" http://localhost:5000
python server.py
pause
