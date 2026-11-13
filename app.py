import copy
import glob
import json
import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.pipeline import TrafficPipeline
from src.database import TrafficDatabase
from src.plate_recognizer import is_valid_tr

# ==============================================================================
# Page Configuration & Modern Design Theme
# ==============================================================================
st.set_page_config(
    page_title="SMART TRAFFIC PRO | Akıllı Trafik İzleme",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Ultra-Modern Glassmorphism & Cyber-Command UI Styles
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
    /* Global Overrides */
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background & Main Container */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(15, 23, 42, 0.95), #060913 85%) !important;
        color: #F1F5F9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(11, 17, 32, 0.85) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #E2E8F0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Header Banner */
    .hero-container {
        position: relative;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .hero-title-group h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-title-group p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 9999px;
        letter-spacing: 1px;
    }

    .live-badge-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #10B981;
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 12px 28px -5px rgba(56, 189, 248, 0.15);
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
    }

    .kpi-card.alert::before {
        background: linear-gradient(90deg, #EF4444, #F87171);
    }

    .kpi-card.success::before {
        background: linear-gradient(90deg, #10B981, #34D399);
    }

    .kpi-card.purple::before {
        background: linear-gradient(90deg, #A855F7, #C084FC);
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 8px;
        letter-spacing: -0.5px;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-unit {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        margin-left: 4px;
    }

    /* Video Player Frame */
    .video-viewport {
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 0 35px -5px rgba(56, 189, 248, 0.15);
        overflow: hidden;
        background: #000;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #94A3B8;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
    }

    /* Evidence Gallery Card */
    .evidence-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.1);
        transition: transform 0.2s;
    }

    .evidence-card:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config(config_path="config/default_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    db = TrafficDatabase(db_path=config.get("output", {}).get("database_path", "output/traffic_surveillance.db"))

    # ==============================================================================
    # Hero Title Banner
    # ==============================================================================
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title-group">
            <h1>⚡ SMART TRAFFIC PRO | Akıllı Trafik İzleme Sistemi</h1>
        </div>
        <div class="live-badge">
            <span class="live-badge-dot"></span>
            SİSTEM AKTİF
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # Sidebar Controls & Hardware Configuration
    # ==============================================================================
    st.sidebar.markdown("### 🎛️ SİSTEM AYARLARI")
    
    speed_limit = st.sidebar.slider(
        "🚨 Hız Sınırı (km/s)",
        min_value=30,
        max_value=140,
        value=config.get("speed", {}).get("speed_limit", 80),
        step=5
    )

    vehicle_conf = st.sidebar.slider(
        "🎯 Araç Tespit Güven Eşiği",
        min_value=0.2,
        max_value=0.9,
        value=float(config.get("model", {}).get("vehicle_conf", 0.40)),
        step=0.05
    )

    device_option = st.sidebar.selectbox(
        "⚡ İşlemci",
        options=["cpu", "auto", "cuda"],
        index=0,
        help="cpu: yalnızca işlemci · auto: NVIDIA GPU varsa onu kullanır · cuda: GPU zorunlu"
    )

    frame_stride = st.sidebar.number_input(
        "⏭️ Kare atlama (NFR-3)", min_value=1, max_value=6,
        value=int(config.get("model", {}).get("frame_stride", 1)),
        help="N>1 ise her N karede bir analiz yapılır. Hız zaman damgasıyla hesaplandığı için doğruluk bozulmaz."
    )

    hud_toggle = st.sidebar.toggle("🖥️ Video üzerinde bilgi paneli (HUD)", value=True)
    blur_toggle = st.sidebar.toggle(
        "🔒 İhlal yapmayan araçların plakasını bulanıklaştır (KVKK)",
        value=bool(config.get("privacy", {}).get("blur_non_violators", True)),
        help="Açık: ekranda ve kaydedilen karelerde yalnızca ihlal yapan aracın plakası görünür.",
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📹 VİDEO KAYNAĞI")
    
    source_type = st.sidebar.radio(
        "Kaynak:",
        options=["Kayıtlı videolar", "Video yükle", "Canlı kamera"],
        index=0
    )

    video_source_path = None
    temp_file = None

    if source_type == "Kayıtlı videolar":
        desktop_dir = os.path.expanduser("~/Desktop")
        sample_videos = []
        for d in ["output/uploads", ".", "speed", "plate", desktop_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".mp4", ".avi", ".mkv")):
                        sample_videos.append(os.path.join(d, f))
        sample_videos = sorted(list(set(sample_videos)))
        if sample_videos:
            video_source_path = st.sidebar.selectbox("Video seçin:", sample_videos)
        else:
            st.sidebar.warning("Kayıtlı video bulunamadı (output/uploads).")
    elif source_type == "Video yükle":
        uploaded_file = st.sidebar.file_uploader("Trafik videosu yükleyin (MP4/AVI/MKV)", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            os.makedirs("output/uploads", exist_ok=True)
            safe_name = "".join(ch for ch in uploaded_file.name if ch.isalnum() or ch in "._-") or "upload.mp4"
            video_source_path = os.path.join("output/uploads", safe_name)
            if not os.path.exists(video_source_path):
                with open(video_source_path, "wb") as out_f:
                    out_f.write(uploaded_file.getbuffer())
    else:
        cam_id = st.sidebar.number_input("Kamera numarası", min_value=0, max_value=5, value=0)
        video_source_path = cam_id

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; color: #94A3B8;">
        <div><strong>Kullanılan modeller:</strong></div>
        <div style="margin-top: 4px;">• YOLO11m (araç tespiti)</div>
        <div>• BoT-SORT (araç takibi)</div>
        <div>• fast-alpr (plaka tespiti + OCR)</div>
        <div>• SQLite veritabanı</div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # Navigation Tabs — one tab per use case (UC-01 … UC-06)
    # ==============================================================================
    tab_cal, tab_live, tab_audit, tab_db, tab_eval, tab_priv = st.tabs([
        "📐 KALİBRASYON",
        "🔴 CANLI İZLEME",
        "🧾 İHLAL İNCELEME",
        "🗄️ KAYITLAR",
        "📊 DEĞERLENDİRME",
        "🔒 GİZLİLİK",
    ])

    # --------------------------------------------------------------------------
    # S1: Camera / road-plane calibration (FR-01)
    # --------------------------------------------------------------------------
    with tab_cal:
        st.markdown("### 📐 Yol düzlemi kalibrasyonu")
        st.caption(
            "Gerçek ölçüsü bilinen bir yol dikdörtgeninin 4 köşesini girin "
            "(ör. iki şerit çizgisi arası = şerit genişliği, yol boyunca kesik çizgi başlangıçları = uzunluk). "
            "Koordinatlar videonun İLK karesine göre 1280×720 ölçeğindedir."
        )
        if not video_source_path or not isinstance(video_source_path, str) or not os.path.exists(str(video_source_path)):
            st.info("Önce soldaki menüden bir video seçin veya yükleyin.")
        else:
            stem = os.path.splitext(os.path.basename(video_source_path))[0]
            if st.session_state.get("cal_msg"):
                st.success(st.session_state.pop("cal_msg"))
            cal_path = os.path.join("config", "calibrations", f"{stem}.json")
            cal = {}
            if os.path.exists(cal_path):
                with open(cal_path, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                st.success(f"Kayıtlı kalibrasyon yüklendi: `{cal_path}`")
            else:
                st.warning("Bu videonun kalibrasyonu yok → hızlar yaklaşık hesaplanır. Doğru hız için kalibre edin.")

            default_pts = cal.get("image_points", [[480, 620], [800, 620], [700, 400], [560, 400]])
            wp = cal.get("world_points", [[0, 0], [3.5, 0], [3.5, 24], [0, 24]])
            c1, c2 = st.columns([2, 1])
            with c2:
                labels = ["1 yakın-sol", "2 yakın-sağ", "3 uzak-sağ", "4 uzak-sol"]
                pts = []
                for k, lab in enumerate(labels):
                    a, b = st.columns(2)
                    x = a.number_input(f"{lab} x", 0.0, 1280.0, float(default_pts[k][0]), 1.0, key=f"cal_x{k}_{stem}")
                    y = b.number_input(f"{lab} y", 0.0, 720.0, float(default_pts[k][1]), 1.0, key=f"cal_y{k}_{stem}")
                    pts.append([x, y])
                width_m = st.number_input("Dikdörtgen genişliği (m) – ör. şerit genişliği", 1.0, 30.0, float(wp[1][0]), 0.05, key=f"cal_w_{stem}")
                length_m = st.number_input("Dikdörtgen uzunluğu (m) – yol boyunca (d_real)", 1.0, 200.0, float(wp[2][1]), 0.5, key=f"cal_l_{stem}")
                cal_limit = st.number_input("Bu kamera için hız sınırı (km/s)", 10, 200, int(cal.get("speed_limit", speed_limit)), 5, key=f"cal_lim_{stem}")
                save_cal = st.button("💾 Kalibrasyonu kaydet", type="primary", use_container_width=True, key=f"cal_save_{stem}")
                reset_cal = st.button("🔄 Kalibrasyonu sıfırla", use_container_width=True, key=f"cal_reset_{stem}",
                                      help="Bu videonun kayıtlı kalibrasyonunu siler ve alanları varsayılana döndürür.")
            with c1:
                cap = cv2.VideoCapture(video_source_path)
                ok, first = cap.read()
                cap.release()
                if ok:
                    img = cv2.resize(first, (1280, 720))
                    world = np.float32([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]])
                    try:
                        Hinv = cv2.getPerspectiveTransform(world, np.float32(pts))
                        for yy in np.arange(-length_m * 0.5, length_m * 2.0, 3.0):
                            line = cv2.perspectiveTransform(np.float32([[[-width_m, yy], [2 * width_m, yy]]]), Hinv)[0]
                            cv2.line(img, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 200, 0), 1)
                    except cv2.error:
                        st.error("Noktalar geçersiz (aynı doğru üzerinde).")
                    cv2.polylines(img, [np.int32(pts).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                    for k, p in enumerate(pts):
                        cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                        cv2.putText(img, str(k + 1), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Turkuaz çizgiler yol boyunca 3 m aralıklı olmalı", use_container_width=True)
            if reset_cal:
                default_path = os.path.join("config", "calibrations", "defaults", f"{stem}.json")
                if os.path.exists(default_path):
                    shutil.copyfile(default_path, cal_path)
                    st.session_state["cal_msg"] = "Kalibrasyon varsayılan değerlere döndürüldü."
                else:
                    if os.path.exists(cal_path):
                        os.remove(cal_path)
                    st.session_state["cal_msg"] = "Kalibrasyon silindi. Bu video artık yaklaşık hız moduyla çalışır."
                for key in [k for k in list(st.session_state.keys()) if str(k).startswith("cal_") and str(k).endswith(f"_{stem}")]:
                    del st.session_state[key]
                st.rerun()
            if save_cal:
                os.makedirs(os.path.dirname(cal_path), exist_ok=True)
                cal.update({
                    "base_resolution": [1280, 720],
                    "image_points": pts,
                    "world_points": [[0.0, 0.0], [width_m, 0.0], [width_m, length_m], [0.0, length_m]],
                    "measure_y_range": cal.get("measure_y_range", [0.45, 0.985]),
                    "speed_limit": int(cal_limit),
                })
                with open(cal_path, "w", encoding="utf-8") as f:
                    json.dump(cal, f, indent=2, ensure_ascii=False)
                st.success(f"Kalibrasyon kaydedildi → `{cal_path}`")

    # --------------------------------------------------------------------------
    # S2: Live surveillance (FR-02, FR-05, FR-10)
    # --------------------------------------------------------------------------
    with tab_live:
        col_video, col_stats = st.columns([7, 3], gap="medium")

        with col_stats:
            st.markdown("#### 📊 ANLIK İSTATİSTİKLER")
            kpi_placeholder = st.empty()
            alert_placeholder = st.empty()

            def render_kpi(total=0, speed_count=0, plate_count=0, viol_count=0, avg_speed=0.0, fps_val=0.0):
                viol_class = "alert" if viol_count > 0 else ""
                kpi_placeholder.markdown(f"""
                <div class="kpi-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="kpi-card success"><div class="kpi-label">🚗 Takip edilen</div><div class="kpi-value">{total}</div></div>
                    <div class="kpi-card"><div class="kpi-label">⚡ Hızı ölçülen</div><div class="kpi-value">{speed_count}</div></div>
                    <div class="kpi-card purple"><div class="kpi-label">🪪 Okunan plaka</div><div class="kpi-value">{plate_count}</div></div>
                    <div class="kpi-card {viol_class}"><div class="kpi-label">🚨 İhlal</div>
                        <div class="kpi-value" style="color: {'#EF4444' if viol_count > 0 else '#F8FAFC'};">{viol_count}</div></div>
                    <div class="kpi-card"><div class="kpi-label">📈 Ort. hız</div><div class="kpi-value">{avg_speed}<span class="kpi-unit">km/s</span></div></div>
                    <div class="kpi-card"><div class="kpi-label">⏱️ FPS</div><div class="kpi-value">{fps_val:.1f}<span class="kpi-unit">fps</span></div></div>
                </div>
                """, unsafe_allow_html=True)

            render_kpi()
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 14px; margin-top: 10px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Aktif ayarlar</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8; margin-top: 2px;">Hız sınırı: {speed_limit} km/s</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">Video: {os.path.basename(str(video_source_path)) if video_source_path is not None else '-'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_video:
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

        b1, b2 = st.columns(2)
        start_btn = b1.button("▶️ BAŞLAT", type="primary", use_container_width=True)
        b2.button("⏹️ DURDUR", use_container_width=True, help="Çalışan analizi durdurur.")

        if start_btn:
            if video_source_path is None:
                st.error("Lütfen geçerli bir video seçin veya yükleyin!")
            else:
                run_cfg = copy.deepcopy(config)
                run_cfg["speed"]["speed_limit"] = speed_limit
                run_cfg["model"]["vehicle_conf"] = vehicle_conf
                run_cfg["model"]["device"] = device_option
                run_cfg["model"]["frame_stride"] = int(frame_stride)
                run_cfg["output"]["hud_stats"] = hud_toggle
                run_cfg.setdefault("privacy", {})["blur_non_violators"] = bool(blur_toggle)

                try:
                    pipeline = TrafficPipeline(config=run_cfg, video_path=str(video_source_path))
                except Exception as e:
                    st.error(f"Sistem başlatılamadı: {e}")
                    st.stop()
                pipeline.violation_manager.start_session(str(video_source_path))
                if pipeline.speed_estimator.H is None:
                    st.warning("Bu videonun kalibrasyonu yok → hızlar yaklaşıktır. Doğru sonuç için KALİBRASYON sekmesini kullanın.")

                cap = cv2.VideoCapture(video_source_path)
                if not cap.isOpened():
                    st.error("Video açılamadı (desteklenmeyen format veya hatalı yol).")
                    st.stop()
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                frame_idx = 0
                t_start = time.time()
                last_viol = 0
                status_text.info("Analiz çalışıyor…")
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1
                        annotated, summary = pipeline.process_frame(frame, frame_idx, fps)

                        if frame_idx % max(1, int(frame_stride)) == 0 or frame_idx == 1:
                            show = annotated
                            if show.shape[1] > 1280:
                                show = cv2.resize(show, (1280, int(show.shape[0] * 1280 / show.shape[1])))
                            video_placeholder.image(cv2.cvtColor(show, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        if total_frames > 0:
                            progress_bar.progress(min(1.0, frame_idx / total_frames))
                        curr_fps = frame_idx / max(0.001, (time.time() - t_start))
                        render_kpi(
                            total=summary.get("total_vehicles", 0),
                            speed_count=summary.get("speed_measured_count", 0),
                            plate_count=summary.get("plate_recognized_count", 0),
                            viol_count=summary.get("violations_count", 0),
                            avg_speed=summary.get("average_speed", 0.0),
                            fps_val=curr_fps
                        )
                        if summary.get("violations_count", 0) > last_viol:   # FR-10 HUD alert
                            last_viol = summary["violations_count"]
                            alert_placeholder.error(f"🚨 HIZ SINIRI AŞILDI — şu ana kadar {last_viol} ihlal")
                finally:
                    cap.release()
                    pipeline.finalize_all()
                    pipeline.violation_manager.export_csv()
                progress_bar.progress(1.0)
                status_text.success(f"Tamamlandı. {frame_idx} kare işlendi · kalibrasyon: {pipeline.calibration_source}")

    # --------------------------------------------------------------------------
    # S3: Audit & correct plates (FR-04, FR-07, FR-09)
    # --------------------------------------------------------------------------
    with tab_audit:
        st.markdown("### 🧾 İhlal inceleme ve plaka düzeltme")
        show_all = st.checkbox("İhlal yapmayan araçların düşük güvenli plakalarını da göster", value=False)
        recs = db.query_records(violations_only=not show_all, limit=500)
        if not recs:
            st.info("İncelenecek kayıt yok.")
        else:
            def _label(r):
                plate = r.get("plate_text") or r.get("plate_candidate") or "—"
                flag = "" if r.get("plate_text") and is_valid_tr(r.get("plate_text") or "") else "  ⚠ geçersiz / düşük güven"
                return f"#{r['id']} · {plate} · {r.get('speed_kmh') or '?'} km/s · {r.get('timestamp')}{flag}"
            choice = st.selectbox("Kayıt", recs, format_func=_label)
            a, b = st.columns([3, 2])
            with a:
                snap = choice.get("snapshot_path")
                if snap and os.path.exists(snap):
                    st.image(snap, caption="Kanıt karesi (diğer araçların plakaları bulanık)", use_container_width=True)
                    plate_img = snap[:-4] + "_plaka.jpg"
                    if os.path.exists(plate_img):
                        st.image(plate_img, caption="Plaka görüntüsü", width=320)
                    with open(snap, "rb") as fh:
                        st.download_button("⬇️ Kanıt fotoğrafını indir", fh, file_name=os.path.basename(snap), mime="image/jpeg")
                else:
                    st.caption("Bu kayıt için kanıt fotoğrafı yok (yalnızca ihlallerde oluşur).")
            with b:
                detected = choice.get("plate_text") or choice.get("plate_candidate") or ""
                st.metric("Hız / sınır", f"{choice.get('speed_kmh') or '?'} / {choice.get('speed_limit')} km/s")
                st.write(f"**Okunan:** `{detected or '—'}`  ·  durum: `{choice.get('plate_status') or '-'}`")
                if detected and not is_valid_tr(detected):
                    st.error("Geçersiz Türk plaka formatı")
                new_plate = st.text_input("Düzeltilmiş plaka", value=detected, key=f"corr_{choice['id']}")
                if st.button("💾 Düzeltmeyi kaydet", type="primary"):
                    ok, msg = db.correct_plate(choice["id"], new_plate, validator=is_valid_tr)
                    (st.success if ok else st.error)(msg)
                log = db.get_corrections(choice["id"])
                if log:
                    st.markdown("**Düzeltme geçmişi (eski → yeni, zaman)**")
                    st.dataframe(pd.DataFrame(log)[["old_plate", "new_plate", "corrected_at"]], hide_index=True)

    # --------------------------------------------------------------------------
    # S4: Records, filters & export (FR-08, FR-11)
    # --------------------------------------------------------------------------
    with tab_db:
        st.markdown("### 🗄️ Geçiş ve ihlal kayıtları")
        f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
        with f1:
            dr = st.date_input("Tarih aralığı", value=(), help="Tüm tarihler için boş bırakın")
        with f2:
            q_plate = st.text_input("Plaka içerir", "", placeholder="34A")
        with f3:
            lanes = ["(tümü)"] + db.get_lanes()
            q_lane = st.selectbox("Şerit", lanes)
        with f4:
            q_viol = st.selectbox("Durum", ["Tüm geçişler", "Yalnızca ihlaller"])
        d_from = d_to = None
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            d_from, d_to = str(dr[0]), str(dr[1])
        elif isinstance(dr, (list, tuple)) and len(dr) == 1:
            d_from = d_to = str(dr[0])
        rows = db.query_records(d_from, d_to, q_plate or None,
                                None if q_lane == "(tümü)" else q_lane, q_viol == "Yalnızca ihlaller")
        if rows:
            df = pd.DataFrame(rows)
            cols = [c for c in ["id", "timestamp", "lane_name", "class_name", "speed_kmh", "speed_limit", "is_speeding",
                                "plate_text", "plate_candidate", "plate_status", "plate_confidence"] if c in df.columns]
            st.dataframe(
                df[cols], use_container_width=True, hide_index=True,
                column_config={
                    "is_speeding": st.column_config.CheckboxColumn("İhlal"),
                    "speed_kmh": st.column_config.NumberColumn("Hız (km/s)", format="%d"),
                    "plate_confidence": st.column_config.ProgressColumn("OCR güveni", min_value=0.0, max_value=1.0, format="%.2f"),
                },
            )
            e1, e2 = st.columns(2)
            e1.download_button("📥 CSV olarak indir", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="traffic_records.csv", mime="text/csv", use_container_width=True)
            if os.path.exists(db.db_path):
                with open(db.db_path, "rb") as fh:
                    e2.download_button("💾 Veritabanını indir", fh, file_name="traffic_surveillance.db",
                                       mime="application/x-sqlite3", use_container_width=True)
        else:
            st.info("Filtrelere uyan kayıt yok.")

    # --------------------------------------------------------------------------
    # S5: Evaluation results (FR-06) — only numbers produced by scripts/
    # --------------------------------------------------------------------------
    with tab_eval:
        st.markdown("### 📊 Değerlendirme sonuçları")
        st.caption("Burada yalnızca `scripts/eval_*.py` betiklerinin ürettiği sonuçlar gösterilir.")
        eval_dir = os.path.join("output", "eval")
        files = sorted(glob.glob(os.path.join(eval_dir, "*.json"))) if os.path.isdir(eval_dir) else []
        if not files:
            st.info("Henüz değerlendirme çalıştırılmadı. Komutlar için README → *Evaluation* bölümüne bakın.")
        for fp in files:
            with open(fp, "r", encoding="utf-8") as fh:
                res = json.load(fh)
            with st.expander(f"{os.path.basename(fp)} · {res.get('created_at', '')}", expanded=True):
                metrics = res.get("metrics", {})
                if metrics:
                    mcols = st.columns(min(4, len(metrics)))
                    for k, (name, val) in enumerate(metrics.items()):
                        mcols[k % len(mcols)].metric(name, val if not isinstance(val, float) else round(val, 3))
                st.json({k: v for k, v in res.items() if k not in ("per_item",)})

    # --------------------------------------------------------------------------
    # S6: Privacy & retention (FR-07, NFR-5)
    # --------------------------------------------------------------------------
    with tab_priv:
        st.markdown("### 🔒 Gizlilik (KVKK) ve veri saklama")
        priv = config.get("privacy", {})
        st.write(f"- İhlal yapmayan araçların plakaları ekranda ve kayıtlarda bulanık: "
                 f"**{'AÇIK' if blur_toggle else 'KAPALI'}** (soldaki 🔒 düğmesiyle değiştirilir)")
        if not blur_toggle:
            st.warning("Bulanıklaştırma KAPALI. Yalnızca açık veri seti videolarıyla kullanın (KVKK).")
        days = st.number_input("İhlal yapmayan geçişlerde plaka saklama süresi (gün)", 1, 365,
                               int(priv.get("retention_days", 30)))
        if st.button("🧹 Saklama süresini şimdi uygula"):
            n = db.apply_retention(int(days))
            st.success(f"{int(days)} günden eski {n} kaydın plakası silindi.")
        st.markdown("---")
        confirm = st.checkbox("Tüm kayıtların kalıcı olarak silineceğini anlıyorum")
        if st.button("🗑️ Tüm kayıtları sil", disabled=not confirm):
            db.clear_records()
            st.success("Tüm kayıtlar silindi.")

if __name__ == "__main__":
    main()
