import os
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.pipeline import TrafficPipeline
from src.database import TrafficDatabase

# ==============================================================================
# Page Configuration & Modern Design Theme
# ==============================================================================
st.set_page_config(
    page_title="ArgusVision | AI Traffic Surveillance SOC",
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
            <h1>⚡ ArgusVision AI | Traffic Surveillance Center</h1>
            <p>SEN499 Graduation Project · Real-Time Multi-Object Tracking (YOLO11), Velocity Estimation & ALPR (fast-alpr)</p>
        </div>
        <div class="live-badge">
            <span class="live-badge-dot"></span>
            ACTIVE SOC SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # Sidebar Controls & Hardware Configuration
    # ==============================================================================
    st.sidebar.markdown("### 🎛️ SYSTEM CONTROLS")
    
    speed_limit = st.sidebar.slider(
        "🚨 Speed Enforcement Limit (km/h)",
        min_value=30,
        max_value=140,
        value=config.get("speed", {}).get("speed_limit", 80),
        step=5
    )

    vehicle_conf = st.sidebar.slider(
        "🎯 Vehicle Detector Confidence",
        min_value=0.2,
        max_value=0.9,
        value=float(config.get("model", {}).get("vehicle_conf", 0.40)),
        step=0.05
    )

    device_option = st.sidebar.selectbox(
        "⚡ Acceleration Engine",
        options=["cpu", "cuda", "0"],
        index=0,
        help="Select CPU or NVIDIA CUDA GPU acceleration"
    )

    hud_toggle = st.sidebar.toggle("🖥️ Overlay Cyber-HUD on Video", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📹 VIDEO SOURCE FEED")
    
    source_type = st.sidebar.radio(
        "Source Mode:",
        options=["Existing Sample Videos", "Upload Video File", "Live Camera Feed"],
        index=0
    )

    video_source_path = None
    temp_file = None

    if source_type == "Existing Sample Videos":
        desktop_dir = os.path.expanduser("~/Desktop")
        sample_videos = []
        for d in [".", "speed", "plate", desktop_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".mp4", ".avi", ".mkv")):
                        sample_videos.append(os.path.join(d, f))
        sample_videos = sorted(list(set(sample_videos)))
        if sample_videos:
            video_source_path = st.sidebar.selectbox("Choose Surveillance Clip:", sample_videos)
        else:
            st.sidebar.warning("No sample video found.")
    elif source_type == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload Traffic Footage (MP4/AVI/MKV)", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.flush()
            video_source_path = temp_file.name
    else:
        cam_id = st.sidebar.number_input("RTSP/Webcam ID", min_value=0, max_value=5, value=0)
        video_source_path = cam_id

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; color: #94A3B8;">
        <div><strong>Models Online:</strong></div>
        <div style="margin-top: 4px;">• YOLO11m (COCO Vehicles)</div>
        <div>• BoT-SORT (Multi-Object MOT)</div>
        <div>• fast-alpr (YOLOv9-t + CCT-S-v2)</div>
        <div>• SQLite3 ACID Storage Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # Navigation Tabs
    # ==============================================================================
    tab_live, tab_db, tab_violations, tab_zones, tab_architecture = st.tabs([
        "🔴 LIVE SURVEILLANCE FEED",
        "🗄️ DATABASE & VEHICLE RECORDS",
        "🚨 VIOLATION EVIDENCE GALLERY",
        "📐 LANE CALIBRATION & POLYGONS",
        "🏛️ SEN499 ARCHITECTURE & SPECS"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Live Surveillance Feed
    # --------------------------------------------------------------------------
    with tab_live:
        col_video, col_stats = st.columns([7, 3], gap="medium")

        with col_stats:
            st.markdown("#### 📊 TELEMETRY & KPIS")
            
            kpi_placeholder = st.empty()
            
            # Initial KPI View
            def render_kpi(total=0, speed_count=0, plate_count=0, viol_count=0, avg_speed=0.0, fps_val=0.0):
                viol_class = "alert" if viol_count > 0 else ""
                kpi_placeholder.markdown(f"""
                <div class="kpi-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="kpi-card success">
                        <div class="kpi-label">🚗 Tracked</div>
                        <div class="kpi-value">{total}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">⚡ Measured</div>
                        <div class="kpi-value">{speed_count}</div>
                    </div>
                    <div class="kpi-card purple">
                        <div class="kpi-label">🪪 Plates</div>
                        <div class="kpi-value">{plate_count}</div>
                    </div>
                    <div class="kpi-card {viol_class}">
                        <div class="kpi-label">🚨 Violations</div>
                        <div class="kpi-value" style="color: {'#EF4444' if viol_count > 0 else '#F8FAFC'};">{viol_count}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">📈 Avg Speed</div>
                        <div class="kpi-value">{avg_speed}<span class="kpi-unit">km/h</span></div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">⏱️ Live FPS</div>
                        <div class="kpi-value">{fps_val:.1f}<span class="kpi-unit">fps</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            render_kpi()

            # Session quick info card
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 14px; margin-top: 10px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Active Enforcement</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8; margin-top: 2px;">Limit: {speed_limit} km/h</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">Camera: {os.path.basename(str(video_source_path)) if video_source_path else 'None'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_video:
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

        start_btn = st.button("▶️ START SURVEILLANCE PIPELINE", type="primary", use_container_width=True)

        if start_btn:
            if video_source_path is None:
                st.error("Please provide or upload a valid surveillance video!")
            else:
                run_cfg = config.copy()
                run_cfg["speed"]["speed_limit"] = speed_limit
                run_cfg["model"]["vehicle_conf"] = vehicle_conf
                run_cfg["model"]["device"] = device_option
                run_cfg["output"]["hud_stats"] = hud_toggle

                pipeline = TrafficPipeline(config=run_cfg)
                pipeline.violation_manager.start_session(str(video_source_path))

                cap = cv2.VideoCapture(video_source_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                frame_idx = 0
                t_start = time.time()
                status_text.info("Surveillance pipeline active · Neural network processing frames...")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    annotated, summary = pipeline.process_frame(frame, frame_idx, fps)

                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

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

                cap.release()
                pipeline.violation_manager.export_csv()
                progress_bar.progress(1.0)
                status_text.success("🎉 Video analysis successfully finalized! Evidence stored in SQLite Database.")

    # --------------------------------------------------------------------------
    # TAB 2: Relational Database & Vehicle Records
    # --------------------------------------------------------------------------
    with tab_db:
        st.markdown("### 🗄️ Relational Database (SQLite) · Vehicle Records")
        records = db.get_all_records()
        
        if records:
            df = pd.DataFrame(records)
            
            # Interactive Filter Bar
            f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
            with f_col1:
                search_plate = st.text_input("🔍 Search License Plate:", "", placeholder="e.g. 34ABC, AD84...")
            with f_col2:
                filter_status = st.selectbox("Vehicle Status:", ["All Records", "Violations Only (Speeding)", "Plates Recognized"])
            with f_col3:
                sort_order = st.selectbox("Sort Order:", ["Newest First", "Fastest Vehicle First"])

            if search_plate:
                df = df[df["plate_text"].astype(str).str.contains(search_plate.upper(), na=False)]
            
            if filter_status == "Violations Only (Speeding)":
                df = df[df["is_speeding"] == 1]
            elif filter_status == "Plates Recognized":
                df = df[df["plate_text"].notna() & (df["plate_text"] != "")]

            if sort_order == "Fastest Vehicle First":
                df = df.sort_values(by="speed_kmh", ascending=False)
            else:
                df = df.sort_values(by="id", ascending=False)

            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "is_speeding": st.column_config.CheckboxColumn("Speeding?", help="True if speed exceeded limit"),
                    "speed_kmh": st.column_config.NumberColumn("Speed (km/h)", format="%d km/h"),
                    "plate_confidence": st.column_config.ProgressColumn("OCR Conf", min_value=0.0, max_value=1.0, format="%.2f"),
                }
            )

            # Export actions
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="📥 Export Current Filtered Table as CSV",
                    data=df.to_csv(index=False).encode('utf-8-sig'),
                    file_name="traffic_surveillance_records.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_col2:
                if os.path.exists(db.db_path):
                    with open(db.db_path, "rb") as f:
                        st.download_button(
                            label="💾 Download Raw SQLite Database (.db)",
                            data=f,
                            file_name="traffic_surveillance.db",
                            mime="application/x-sqlite3",
                            use_container_width=True
                        )
        else:
            st.info("No records in database yet. Launch a surveillance video stream above to populate data.")

    # --------------------------------------------------------------------------
    # TAB 3: Violation Evidence Gallery
    # --------------------------------------------------------------------------
    with tab_violations:
        st.markdown("### 🚨 Photographic Violation Evidence Gallery")
        st.markdown("High-resolution cropped vehicle snapshots captured automatically at the moment of speed limit violation.")
        
        viol_dir = config.get("output", {}).get("violations_dir", "output/ihlaller")
        if os.path.exists(viol_dir):
            images = [os.path.join(viol_dir, f) for f in os.listdir(viol_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if images:
                cols = st.columns(3)
                for idx, img_path in enumerate(images):
                    col = cols[idx % 3]
                    with col:
                        st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                        with open(img_path, "rb") as img_file:
                            st.download_button(
                                label="⬇️ Download Evidence JPEG",
                                data=img_file,
                                file_name=os.path.basename(img_path),
                                mime="image/jpeg",
                                key=f"dl_snap_{idx}",
                                use_container_width=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("Tebrikler! No speed limit violations detected in the latest session.")
        else:
            st.info("No violation records stored yet.")

    # --------------------------------------------------------------------------
    # TAB 4: Lane Calibration & Polygons
    # --------------------------------------------------------------------------
    with tab_zones:
        st.markdown("### 📐 Road Lane & Speed Calibration Zones")
        st.markdown("Zone polygons and calibrated distance ratios defined in `config/default_config.yaml`:")
        
        col_z1, col_z2 = st.columns([1, 1])
        with col_z1:
            st.markdown("#### Active Zone Polygons")
            st.json(config.get("speed", {}).get("zones", []))
        
        with col_z2:
            st.markdown("""
            #### 🛠️ Interactive Calibration Utility
            To calibrate new road camera angles or highway lanes interactively with mouse clicks:
            ```bash
            python zone_calibrator.py --input video.mp4
            ```
            - **Left Click:** Add polygon vertex.
            - **Right Click:** Undo last point.
            - **'s' Key:** Save active polygon.
            - **'q' Key:** Export formatted JSON coordinates ready to paste into `config/default_config.yaml`.
            """)

    # --------------------------------------------------------------------------
    # TAB 5: SEN499 Architecture & Specs
    # --------------------------------------------------------------------------
    with tab_architecture:
        st.markdown("""
        ### 🏛️ SEN499 Graduation Project Specifications & Architectural Verification
        
        #### Section 3.1 Compliance: 6 Core Use Cases (Fully Implemented)
        1. **UC-1: Multi-Class Vehicle Detection & Continuous Tracking**  
           Deep neural network inference using **YOLO11m** with **BoT-SORT** persistent ID preservation across visual occlusions.
        2. **UC-2: Frame-Accurate Hardware-Independent Speed Estimation**  
           Velocity computation based on fixed camera framerate intervals $\\Delta t = \\frac{\\Delta \\text{Frames}}{\\text{FPS}}$, immune to processing computer slowdown.
        3. **UC-3: Deep Learning License Plate Recognition (ALPR)**  
           Two-stage neural plate detection with **YOLOv9-t** and optical character recognition with Compact Convolutional Transformers (**CCT-S-v2**).
        4. **UC-4: Multi-Frame Candidate Accumulation & Confidence Locking**  
           Temporal candidate buffer collecting $N$ valid plate reads and locking the highest statistical confidence to eliminate frame jitter.
        5. **UC-5: Speeding Enforcement & High-Resolution Evidence Capture**  
           Automated threshold comparator saving annotated photographic evidence snapshots into `output/ihlaller/`.
        6. **UC-6: Persistent Relational Storage & Analytical Reporting**  
           ACID-compliant **SQLite3** relational persistence with indexed queries, CSV export, and Streamlit telemetry.
        """)


if __name__ == "__main__":
    main()
