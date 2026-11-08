/**
 * ArgusVision | AI Traffic Surveillance SOC Frontend Logic (Vanilla JS)
 */

let statsInterval = null;
let recordsInterval = null;
let isStreaming = false;

document.addEventListener("DOMContentLoaded", () => {
    // 1. Saat Başlatma
    startClock();

    // 2. İlk Yüklemeler
    loadRecords();
    loadGallery();
    startTelemetryPolling();

    // 3. Drag and Drop Dosya Yükleme Olayları
    setupDropzone();
});

// ------------------------------------------------------------------------------
// Saat ve Canlı Durum
// ------------------------------------------------------------------------------
function startClock() {
    function update() {
        const now = new Date();
        const str = now.toTimeString().split(" ")[0] + " · " + now.toLocaleDateString();
        document.getElementById("clockDisplay").textContent = str;
    }
    update();
    setInterval(update, 1000);
}

function updateSpeedLabel(val) {
    document.getElementById("speedLimitVal").textContent = val + " km/h";
}

// ------------------------------------------------------------------------------
// Stream Kontrolleri (Start, Pause, Stop)
// ------------------------------------------------------------------------------
async function startStream() {
    const videoPath = document.getElementById("videoSourceInput").value.trim();
    const speedLimit = parseInt(document.getElementById("speedLimitInput").value, 10);

    try {
        const res = await fetch("/api/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                action: "start",
                video_path: videoPath,
                speed_limit: speedLimit
            })
        });
        const data = await res.json();
        
        if (data.status === "started") {
            isStreaming = true;
            document.getElementById("placeholderOverlay").style.display = "none";
            document.getElementById("streamPlayer").src = "/video_feed?" + new Date().getTime();
            document.getElementById("systemStatusText").textContent = "ANALYSIS IN PROGRESS";
            document.getElementById("systemDot").style.background = "#00F2FE";
            document.getElementById("systemDot").style.boxShadow = "0 0 10px #00F2FE";
        }
    } catch (err) {
        console.error("Error starting stream:", err);
        alert("Failed to start surveillance stream. Please verify server status.");
    }
}

async function pauseStream() {
    try {
        const res = await fetch("/api/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "pause" })
        });
        const data = await res.json();
        document.getElementById("btnPause").innerHTML = data.is_paused 
            ? '<i class="fa-solid fa-play"></i> RESUME' 
            : '<i class="fa-solid fa-pause"></i> PAUSE';
    } catch (err) {
        console.error("Error pausing stream:", err);
    }
}

async function stopStream() {
    try {
        await fetch("/api/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "stop" })
        });
        isStreaming = false;
        document.getElementById("placeholderOverlay").style.display = "flex";
        document.getElementById("systemStatusText").textContent = "STANDBY / COMPLETED";
        document.getElementById("systemDot").style.background = "#10B981";
        document.getElementById("systemDot").style.boxShadow = "0 0 8px #10B981";
        loadRecords();
        loadGallery();
    } catch (err) {
        console.error("Error stopping stream:", err);
    }
}

function toggleFullscreen() {
    const el = document.getElementById("viewportContainer");
    if (!document.fullscreenElement) {
        el.requestFullscreen().catch(err => alert("Fullscreen error: " + err.message));
    } else {
        document.exitFullscreen();
    }
}

// ------------------------------------------------------------------------------
// Canlı Telemetri Polling
// ------------------------------------------------------------------------------
function startTelemetryPolling() {
    statsInterval = setInterval(async () => {
        try {
            const res = await fetch("/api/stats");
            const stats = await res.json();

            // KPI Değerlerini Güncelle
            document.getElementById("kpiTotal").textContent = stats.total_vehicles || 0;
            document.getElementById("kpiSpeed").textContent = stats.speed_measured_count || 0;
            document.getElementById("kpiPlates").textContent = stats.plate_recognized_count || 0;
            
            const violCount = stats.violations_count || 0;
            document.getElementById("kpiViolations").textContent = violCount;
            if (violCount > 0) {
                document.getElementById("cardViolations").classList.add("danger");
                document.getElementById("violationTag").textContent = "CRITICAL ALERT";
            }

            document.getElementById("kpiAvgSpeed").innerHTML = `${stats.average_speed || 0.0} <span class="unit">km/h</span>`;
            document.getElementById("kpiFps").innerHTML = `${stats.fps || 0.0} <span class="unit">FPS</span>`;
            document.getElementById("feedFpsBadge").textContent = `${stats.fps || 0.0} FPS`;

            if (stats.is_running && !isStreaming) {
                isStreaming = true;
                document.getElementById("placeholderOverlay").style.display = "none";
            }
        } catch (err) {
            // Polling hatasını sessizce yut
        }
    }, 600);

    // Tabloyu da periyodik güncelle
    setInterval(() => {
        if (isStreaming) {
            loadRecords();
            loadGallery();
        }
    }, 3000);
}

// ------------------------------------------------------------------------------
// Sekme Değişimi
// ------------------------------------------------------------------------------
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    const btn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick").includes(tabId));
    if (btn) btn.classList.add("active");

    const content = document.getElementById(tabId);
    if (content) content.classList.add("active");

    if (tabId === "tab-db") loadRecords();
    if (tabId === "tab-gallery") loadGallery();
}

// ------------------------------------------------------------------------------
// Veritabanı Kayıtları Tablosu
// ------------------------------------------------------------------------------
async function loadRecords() {
    const plateQuery = document.getElementById("plateSearch").value.trim();
    const onlyViolations = document.getElementById("checkViolationsOnly").checked;

    try {
        const url = `/api/records?plate=${encodeURIComponent(plateQuery)}&violations_only=${onlyViolations}`;
        const res = await fetch(url);
        const records = await res.json();

        const tbody = document.getElementById("recordsTableBody");
        if (!records || records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No vehicle records found.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => {
            const isSpeeding = r.is_speeding === 1;
            const speedDisplay = r.speed_kmh ? `${r.speed_kmh} km/h` : `<span style="color:#64748B;">Pending</span>`;
            const plateDisplay = r.plate_text 
                ? `<span class="plate-badge">${r.plate_text}</span>` 
                : `<span style="color:#64748B;">Scanning...</span>`;
            const statusDisplay = isSpeeding 
                ? `<span class="status-pill viol">VIOLATION</span>` 
                : `<span class="status-pill ok">COMPLIANT</span>`;

            return `
            <tr>
                <td><strong>#${r.track_id}</strong></td>
                <td><span style="text-transform:uppercase;">${r.class_name || 'car'}</span></td>
                <td><strong>${speedDisplay}</strong></td>
                <td>${plateDisplay}</td>
                <td>${statusDisplay}</td>
                <td><small style="font-family:monospace; color:#94A3B8;">${r.timestamp.split(" ")[1] || r.timestamp}</small></td>
            </tr>
            `;
        }).join("");

    } catch (err) {
        console.error("Error loading records:", err);
    }
}

function filterRecords() {
    loadRecords();
}

// ------------------------------------------------------------------------------
// İhlal Fotoğrafları Galerisi (Evidence Gallery)
// ------------------------------------------------------------------------------
async function loadGallery() {
    try {
        const res = await fetch("/api/violations");
        const images = await res.json();
        const grid = document.getElementById("galleryGrid");

        if (!images || images.length === 0) {
            grid.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;">No violation evidence recorded yet.</div>`;
            return;
        }

        grid.innerHTML = images.map(img => `
            <div class="gallery-card" onclick="openModal('${img.url}', '${img.filename}')">
                <img src="${img.url}" alt="${img.filename}" loading="lazy">
                <div class="gallery-info">
                    <div><strong>${img.filename.replace("violation_", "").replace(".jpg", "")}</strong></div>
                    <small>Click to inspect evidence</small>
                </div>
            </div>
        `).join("");

    } catch (err) {
        console.error("Error loading gallery:", err);
    }
}

// ------------------------------------------------------------------------------
// Modal Lightbox
// ------------------------------------------------------------------------------
function openModal(imgUrl, caption) {
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImg");
    const modalCaption = document.getElementById("modalCaption");

    modal.style.display = "flex";
    modalImg.src = imgUrl;
    modalCaption.textContent = caption;
}

function closeModal() {
    document.getElementById("imageModal").style.display = "none";
}

// ------------------------------------------------------------------------------
// Dosya Sürükle & Bırak (Drag & Drop Upload)
// ------------------------------------------------------------------------------
function setupDropzone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#00F2FE";
        dropZone.style.background = "rgba(0, 242, 254, 0.08)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "rgba(56, 189, 248, 0.3)";
        dropZone.style.background = "rgba(15, 23, 42, 0.5)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "rgba(56, 189, 248, 0.3)";
        dropZone.style.background = "rgba(15, 23, 42, 0.5)";
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files);
        }
    });
}

async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];

    const formData = new FormData();
    formData.append("video", file);

    const progressBar = document.getElementById("uploadProgress");
    const progressFill = document.getElementById("progressFill");
    progressBar.style.display = "block";
    progressFill.style.width = "30%";

    try {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/upload", true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = percent + "%";
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                document.getElementById("videoSourceInput").value = data.filepath;
                progressBar.style.display = "none";
                alert(`✅ Video successfully uploaded!\nSource updated to: ${data.filepath}`);
            } else {
                alert("Upload failed. Server returned status: " + xhr.status);
                progressBar.style.display = "none";
            }
        };

        xhr.send(formData);

    } catch (err) {
        console.error("Upload error:", err);
        alert("Upload error occurred.");
        progressBar.style.display = "none";
    }
}
