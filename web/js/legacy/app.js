/**
 * SMART TRAFFIC PRO | Yapay Zeka Trafik İzleme SOC Arayüz Mantığı (Vanilla JS)
 */

let statsInterval = null;
let isStreaming = false;
let dismissedAlertKey = null;

document.addEventListener("DOMContentLoaded", () => {
    // 1. Saat Başlatma
    startClock();

    // 2. Sunucudan varsayılan değerleri al
    loadInitialConfig();

    // 3. İlk Yüklemeler
    loadRecords();
    loadGallery();
    startTelemetryPolling();

    // 4. Dosya Sürükle & Bırak Yükleme
    setupDropzone();

    // 5. Sayfa yenilendiğinde süren toplu analiz varsa ilerlemeyi göstermeye devam et
    resumeBatchIfRunning();
});

// ------------------------------------------------------------------------------
// Yardımcılar
// ------------------------------------------------------------------------------
function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

async function readJson(res) {
    try {
        return await res.json();
    } catch (e) {
        return {};
    }
}

async function loadInitialConfig() {
    try {
        const res = await fetch("/api/config");
        const cfg = await readJson(res);
        const input = document.getElementById("videoSourceInput");
        if (cfg.default_video_path && input && !input.dataset.userEdited) {
            input.value = cfg.default_video_path;
        }
        if (cfg.speed_limit) {
            document.getElementById("speedLimitInput").value = cfg.speed_limit;
            updateSpeedLabel(cfg.speed_limit);
        }
    } catch (e) {
        // Varsayılanlar HTML'de mevcut
    }
}

// ------------------------------------------------------------------------------
// Saat ve Canlı Durum
// ------------------------------------------------------------------------------
function startClock() {
    function update() {
        const now = new Date();
        const str = now.toTimeString().split(" ")[0] + " · " + now.toLocaleDateString("tr-TR");
        document.getElementById("clockDisplay").textContent = str;
    }
    update();
    setInterval(update, 1000);
}

function updateSpeedLabel(val) {
    document.getElementById("speedLimitVal").textContent = val + " km/s";
}

function setSystemStatus(text, color) {
    document.getElementById("systemStatusText").textContent = text;
    const dot = document.getElementById("systemDot");
    dot.style.background = color;
    dot.style.boxShadow = `0 0 10px ${color}`;
}

function showLiveView() {
    const finishedPlayer = document.getElementById("finishedVideoPlayer");
    finishedPlayer.pause();
    finishedPlayer.style.display = "none";
    document.getElementById("streamPlayer").style.display = "block";
    document.getElementById("btnDownloadProcessed").style.display = "none";
}

// ------------------------------------------------------------------------------
// Akış Kontrolleri (Başlat, Duraklat, Durdur)
// ------------------------------------------------------------------------------
async function startStream() {
    const videoPath = document.getElementById("videoSourceInput").value.trim();
    const speedLimit = parseInt(document.getElementById("speedLimitInput").value, 10);
    const btn = document.getElementById("btnStart");

    btn.disabled = true;
    setSystemStatus("MODEL YÜKLENİYOR...", "#F59E0B");

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
        const data = await readJson(res);

        if (res.ok && data.status === "started") {
            isStreaming = true;
            showLiveView();
            document.getElementById("placeholderOverlay").style.display = "none";
            document.getElementById("streamPlayer").src = "/video_feed?" + Date.now();
            setSystemStatus("ANALİZ SÜRÜYOR", "#00F2FE");
            loadRecords();
        } else {
            setSystemStatus("SİSTEM HAZIR", "#10B981");
            alert("Video akışı başlatılamadı:\n" + (data.error || `Sunucu durum kodu: ${res.status}`));
        }
    } catch (err) {
        console.error("Akış başlatma hatası:", err);
        setSystemStatus("BAĞLANTI HATASI", "#EF4444");
        alert("Video akışı başlatılamadı. Lütfen sunucunun çalıştığını doğrulayın.");
    } finally {
        btn.disabled = false;
    }
}

async function pauseStream() {
    try {
        const res = await fetch("/api/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "pause" })
        });
        const data = await readJson(res);
        const btnPause = document.getElementById("btnPause");
        if (btnPause) {
            btnPause.innerHTML = data.is_paused
                ? '<i class="fa-solid fa-play"></i> DEVAM ET'
                : '<i class="fa-solid fa-pause"></i> DURAKLAT';
        }
    } catch (err) {
        console.error("Duraklatma hatası:", err);
    }
}

function onStreamFinished(statusText) {
    isStreaming = false;
    document.getElementById("placeholderOverlay").style.display = "flex";
    setSystemStatus(statusText || "BEKLEMEDE / TAMAMLANDI", "#10B981");
    loadRecords();
    loadGallery();
    if (document.getElementById("tab-charts")?.classList.contains("active")) {
        loadAnalyticsCharts();
    }
}

async function stopStream() {
    try {
        await fetch("/api/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "stop" })
        });
    } catch (err) {
        console.error("Durdurma hatası:", err);
    }
    onStreamFinished("BEKLEMEDE / DURDURULDU");
}

/**
 * DURDUR butonu: canlı akışı, toplu analizi ve oynatıcıyı tamamen durdurur.
 */
async function stopEverything() {
    // 1. Toplu analizi iptal et
    if (batchPollInterval) {
        clearInterval(batchPollInterval);
        batchPollInterval = null;
    }
    try {
        await fetch("/api/batch/cancel", { method: "POST" });
    } catch (err) {
        console.error("Toplu analiz iptal hatası:", err);
    }
    document.getElementById("batchProgressOverlay").style.display = "none";

    // 2. Oynatıcıyı durdur
    showLiveView();
    document.getElementById("streamPlayer").src = "";

    // 3. Canlı akışı durdur
    await stopStream();
}

function toggleFullscreen() {
    const el = document.getElementById("viewportContainer");
    if (!document.fullscreenElement) {
        el.requestFullscreen().catch(err => alert("Tam ekran hatası: " + err.message));
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
                document.getElementById("violationTag").textContent = "KRİTİK UYARI";
            } else {
                document.getElementById("cardViolations").classList.remove("danger");
                document.getElementById("violationTag").textContent = "UYGULANIYOR";
            }

            document.getElementById("kpiAvgSpeed").innerHTML = `${stats.average_speed || 0.0} <span class="unit">km/s</span>`;
            document.getElementById("kpiFps").innerHTML = `${stats.fps || 0.0} <span class="unit">FPS</span>`;
            document.getElementById("feedFpsBadge").textContent = `${stats.fps || 0.0} FPS`;

            // Kara Liste Alarmı Kontrolü (kapatılan alarm tekrar açılmaz)
            const alert_ = stats.latest_alert;
            if (alert_) {
                const key = `${alert_.plate}|${alert_.timestamp}`;
                const banner = document.getElementById("watchlistAlertBanner");
                const desc = document.getElementById("watchlistAlertDesc");
                if (banner && desc && key !== dismissedAlertKey) {
                    desc.textContent = `Plaka: ${alert_.plate} - Sebep: ${alert_.reason} (${alert_.timestamp})`;
                    banner.dataset.alertKey = key;
                    banner.style.display = "flex";
                }
            }

            if (stats.is_running && !isStreaming) {
                // Sayfa yenilendiğinde süren akışa yeniden bağlan
                isStreaming = true;
                showLiveView();
                document.getElementById("placeholderOverlay").style.display = "none";
                document.getElementById("streamPlayer").src = "/video_feed?" + Date.now();
                setSystemStatus("ANALİZ SÜRÜYOR", "#00F2FE");
            } else if (!stats.is_running && isStreaming) {
                // Video sona erdi veya hata oluştu
                if (stats.error) {
                    onStreamFinished("HATA");
                    alert("Analiz sırasında hata oluştu:\n" + stats.error);
                } else {
                    onStreamFinished("ANALİZ TAMAMLANDI");
                }
            } else if (stats.is_running) {
                setSystemStatus(stats.is_paused ? "DURAKLATILDI" : "ANALİZ SÜRÜYOR", stats.is_paused ? "#F59E0B" : "#00F2FE");
            }
        } catch (err) {
            // Hata durumunu sessizce geç
        }
    }, 600);

    // Tabloyu ve galeriyi periyodik güncelle
    setInterval(() => {
        if (isStreaming) {
            loadRecords();
            loadGallery();
            if (document.getElementById("tab-charts")?.classList.contains("active")) {
                loadAnalyticsCharts();
            }
        }
    }, 3000);
}

function dismissWatchlistAlert() {
    const banner = document.getElementById("watchlistAlertBanner");
    if (banner) {
        dismissedAlertKey = banner.dataset.alertKey || null;
        banner.style.display = "none";
    }
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
    if (tabId === "tab-charts") loadAnalyticsCharts();
    if (tabId === "tab-watchlist") loadWatchlist();
}

// ------------------------------------------------------------------------------
// Veritabanı Kayıtları Tablosu & Temizleme
// ------------------------------------------------------------------------------
const CLASS_TR = {
    car: "Otomobil",
    truck: "Kamyon",
    bus: "Otobüs",
    motorcycle: "Motosiklet"
};

async function loadRecords() {
    const plateQuery = document.getElementById("plateSearch").value.trim();
    const onlyViolations = document.getElementById("checkViolationsOnly").checked;

    try {
        const url = `/api/records?plate=${encodeURIComponent(plateQuery)}&violations_only=${onlyViolations}`;
        const res = await fetch(url);
        const records = await res.json();

        const tbody = document.getElementById("recordsTableBody");
        if (!Array.isArray(records) || records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="empty-state">Kayıtlı araç verisi bulunamadı.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => {
            const isSpeeding = Boolean(r.is_speeding);
            const speedDisplay = r.speed_kmh ? `${escapeHtml(r.speed_kmh)} km/s` : `<span style="color:#64748B;">Bekliyor</span>`;

            let plateDisplay = `<span style="color:#64748B;">Taranıyor...</span>`;
            if (r.plate_text) {
                plateDisplay = `<span class="plate-badge">${escapeHtml(r.plate_text)}</span>`;
                if (r.watchlist_match) {
                    plateDisplay += ` <span class="badge-blacklist" title="Arama Nedeni: ${escapeHtml(r.watchlist_match.reason)}">🚨 ARANAN</span>`;
                }
            }

            const statusDisplay = isSpeeding
                ? `<span class="status-pill viol">HIZ İHLALİ</span>`
                : `<span class="status-pill ok">KURALLARA UYGUN</span>`;
            const trClass = CLASS_TR[(r.class_name || "").toLowerCase()] || (r.class_name || "Araç");

            const pdfBtn = isSpeeding
                ? `<a href="/api/records/${r.id}/pdf" class="btn-pdf" target="_blank" title="Resmi PDF İhlal Tutanağı İndir"><i class="fa-solid fa-file-pdf"></i> Tutanak</a>`
                : `<a href="/api/records/${r.id}/pdf" class="btn-pdf" target="_blank" style="border-color:#475569; color:#94A3B8;" title="PDF Araç Bilgi Formu"><i class="fa-solid fa-file-lines"></i> Rapor</a>`;

            const rowStyle = r.watchlist_match ? 'style="background:rgba(220, 38, 38, 0.12);"' : '';
            const timeText = (r.timestamp || "").split(" ")[1] || r.timestamp || "";

            return `
            <tr ${rowStyle}>
                <td><strong>#${escapeHtml(r.track_id)}</strong></td>
                <td><span>${escapeHtml(trClass)}</span></td>
                <td><strong>${speedDisplay}</strong></td>
                <td>${plateDisplay}</td>
                <td>${statusDisplay}</td>
                <td><small style="font-family:monospace; color:#94A3B8;">${escapeHtml(timeText)}</small></td>
                <td>${pdfBtn}</td>
            </tr>
            `;
        }).join("");

    } catch (err) {
        console.error("Kayıtlar yüklenirken hata:", err);
    }
}

function filterRecords() {
    loadRecords();
}

/**
 * Kayıtlar ekranını ve veritabanını temizleme fonksiyonu
 */
async function clearAllRecords() {
    const onay = confirm("Tüm araç ve ihlal kayıtları temizlensin mi?\nBu işlem mevcut kayıtları sıfırlayacaktır.");
    if (!onay) return;

    try {
        const res = await fetch("/api/records/clear", { method: "POST" });
        const data = await readJson(res);
        if (data.status === "cleared") {
            loadRecords();
            loadGallery();
            if (document.getElementById("tab-charts")?.classList.contains("active")) {
                loadAnalyticsCharts();
            }
            document.getElementById("kpiTotal").textContent = "0";
            document.getElementById("kpiSpeed").textContent = "0";
            document.getElementById("kpiPlates").textContent = "0";
            document.getElementById("kpiViolations").textContent = "0";
            document.getElementById("kpiAvgSpeed").innerHTML = `0.0 <span class="unit">km/s</span>`;
            dismissWatchlistAlert();
            alert("✅ Tüm kayıtlar başarıyla temizlendi.");
        } else {
            alert("Kayıtlar temizlenirken bir hata oluştu.");
        }
    } catch (err) {
        console.error("Temizleme hatası:", err);
        alert("Sunucu ile bağlantı kurulamadı.");
    }
}

// ------------------------------------------------------------------------------
// İhlal Fotoğrafları Galerisi (Evidence Gallery)
// ------------------------------------------------------------------------------
async function loadGallery() {
    try {
        const res = await fetch("/api/violations");
        const images = await res.json();
        const grid = document.getElementById("galleryGrid");

        if (!Array.isArray(images) || images.length === 0) {
            grid.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;">Henüz kaydedilmiş ihlal kanıtı bulunmuyor.</div>`;
            return;
        }

        grid.innerHTML = images.map(img => `
            <div class="gallery-card" data-url="${escapeHtml(img.url)}" data-filename="${escapeHtml(img.filename)}">
                <img src="${escapeHtml(img.url)}" alt="${escapeHtml(img.filename)}" loading="lazy">
                <div class="gallery-info">
                    <div><strong>${escapeHtml(img.filename.replace("violation_", "").replace(/\.(jpg|jpeg|png)$/i, ""))}</strong></div>
                    <small>Kanıtı incelemek için tıklayın</small>
                </div>
            </div>
        `).join("");

        grid.querySelectorAll(".gallery-card").forEach(card => {
            card.addEventListener("click", () => openModal(card.dataset.url, card.dataset.filename));
        });

    } catch (err) {
        console.error("Galeri yükleme hatası:", err);
    }
}

// ------------------------------------------------------------------------------
// Modal Lightbox & PDF İndirme
// ------------------------------------------------------------------------------
function openModal(imgUrl, caption) {
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImg");
    const modalCaption = document.getElementById("modalCaption");
    const modalPdfBtn = document.getElementById("modalPdfBtn");

    modal.style.display = "flex";
    modalImg.src = imgUrl;
    modalCaption.textContent = caption;
    if (modalPdfBtn) {
        modalPdfBtn.href = `/api/violations/${encodeURIComponent(caption)}/pdf`;
    }
}

function closeModal() {
    document.getElementById("imageModal").style.display = "none";
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
});

// ------------------------------------------------------------------------------
// Kara Liste (Aranan Araç) Yönetimi
// ------------------------------------------------------------------------------
async function loadWatchlist() {
    try {
        const res = await fetch("/api/watchlist");
        const list = await res.json();
        const tbody = document.getElementById("watchlistTableBody");
        if (!Array.isArray(list) || list.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="empty-state">Kayıtlı kara liste plakası bulunmuyor.</td></tr>`;
            return;
        }

        tbody.innerHTML = list.map(item => `
            <tr>
                <td><span class="plate-badge" style="border-color:#EF4444; color:#FCA5A5;">${escapeHtml(item.plate_text)}</span></td>
                <td><strong>${escapeHtml(item.reason)}</strong></td>
                <td><span class="status-pill viol">${item.severity === 'danger' ? 'Kritik Alarm' : 'İnceleme'}</span></td>
                <td><small style="font-family:monospace; color:#94A3B8;">${escapeHtml(item.added_at)}</small></td>
                <td>
                    <button class="btn btn-sm btn-danger" onclick="deleteWatchlistPlate(${Number(item.id)})" title="Listeden Çıkar">
                        <i class="fa-solid fa-trash"></i> Sil
                    </button>
                </td>
            </tr>
        `).join("");
    } catch (err) {
        console.error("Kara liste yükleme hatası:", err);
    }
}

async function addWatchlistPlate() {
    const plateInput = document.getElementById("watchPlateInput");
    const reasonInput = document.getElementById("watchReasonInput");
    const severityInput = document.getElementById("watchSeverityInput");

    const plate_text = plateInput.value.trim().toUpperCase();
    const reason = reasonInput.value.trim();
    const severity = severityInput.value;

    if (!plate_text) {
        alert("Lütfen bir plaka girin (örn: 34VF5821)");
        return;
    }
    if (!reason) {
        alert("Lütfen aranma / izleme sebebi girin");
        return;
    }

    try {
        const res = await fetch("/api/watchlist", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ plate_text, reason, severity })
        });
        const data = await readJson(res);
        if (data.status === "added") {
            plateInput.value = "";
            reasonInput.value = "";
            loadWatchlist();
            loadRecords();
        } else {
            alert("Plaka eklenemedi." + (data.error ? "\n" + data.error : ""));
        }
    } catch (err) {
        console.error("Kara liste ekleme hatası:", err);
        alert("Sunucu ile bağlantı kurulamadı.");
    }
}

async function deleteWatchlistPlate(itemId) {
    if (!confirm("Bu plaka kara listeden çıkarılsın mı?")) return;
    try {
        const res = await fetch(`/api/watchlist/${itemId}`, { method: "DELETE" });
        const data = await readJson(res);
        if (data.status === "deleted") {
            loadWatchlist();
            loadRecords();
        } else {
            alert("Plaka silinemedi.");
        }
    } catch (err) {
        console.error("Silme hatası:", err);
    }
}

// ------------------------------------------------------------------------------
// Trafik İstatistikleri ve Grafikler (Chart.js)
// ------------------------------------------------------------------------------
let speedChartInstance = null;
let classChartInstance = null;

async function loadAnalyticsCharts() {
    try {
        const res = await fetch("/api/stats/analytics");
        const data = await res.json();

        // Mini istatistik sayaçları
        const elTotal = document.getElementById("chartTotalCount");
        const elRate = document.getElementById("chartViolRate");
        const elAvg = document.getElementById("chartAvgSpeed");
        const elMax = document.getElementById("chartMaxSpeed");

        if (elTotal) elTotal.textContent = data.total_vehicles || 0;
        if (elRate) elRate.textContent = `%${data.violation_rate || 0.0}`;
        if (elAvg) elAvg.textContent = `${data.average_speed || 0.0} km/s`;
        if (elMax) elMax.textContent = `${data.max_speed || 0} km/s`;

        if (typeof Chart === "undefined") {
            console.warn("Chart.js yüklenemedi (internet bağlantısı gerekli).");
            return;
        }

        // 1. Hız Dağılımı Çubuk Grafiği
        const speedCanvas = document.getElementById("speedChart");
        if (speedCanvas) {
            const speedCtx = speedCanvas.getContext("2d");
            const speedDist = data.speed_distribution || {};
            const speedLabels = Object.keys(speedDist);
            const speedValues = Object.values(speedDist);

            const barColors = speedLabels.map(l => {
                if (l.includes("81-100") || l.includes("100+")) return "rgba(239, 68, 68, 0.85)"; // Kırmızı - İhlal
                if (l.includes("71-80")) return "rgba(245, 158, 11, 0.85)"; // Amber - Sınır
                return "rgba(56, 189, 248, 0.85)"; // Camgöbeği - Kurallara uygun
            });

            if (speedChartInstance) {
                speedChartInstance.data.labels = speedLabels;
                speedChartInstance.data.datasets[0].data = speedValues;
                speedChartInstance.data.datasets[0].backgroundColor = barColors;
                speedChartInstance.update();
            } else {
                speedChartInstance = new Chart(speedCtx, {
                    type: "bar",
                    data: {
                        labels: speedLabels,
                        datasets: [{
                            label: "Araç Sayısı",
                            data: speedValues,
                            backgroundColor: barColors,
                            borderRadius: 6,
                            borderWidth: 1,
                            borderColor: "rgba(255, 255, 255, 0.15)"
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            x: {
                                ticks: { color: "#94A3B8", font: { size: 10 } },
                                grid: { color: "rgba(255, 255, 255, 0.05)" }
                            },
                            y: {
                                beginAtZero: true,
                                ticks: { color: "#94A3B8", stepSize: 1, font: { size: 10 } },
                                grid: { color: "rgba(255, 255, 255, 0.05)" }
                            }
                        }
                    }
                });
            }
        }

        // 2. Araç Sınıfı Halka Grafiği
        const classCanvas = document.getElementById("classChart");
        if (classCanvas) {
            const classCtx = classCanvas.getContext("2d");
            const classDist = data.class_distribution || {};
            const classLabels = Object.keys(classDist);
            const classValues = Object.values(classDist);

            if (classChartInstance) {
                classChartInstance.data.labels = classLabels;
                classChartInstance.data.datasets[0].data = classValues;
                classChartInstance.update();
            } else {
                classChartInstance = new Chart(classCtx, {
                    type: "doughnut",
                    data: {
                        labels: classLabels,
                        datasets: [{
                            data: classValues,
                            backgroundColor: [
                                "rgba(56, 189, 248, 0.85)",  // Otomobil (Mavi)
                                "rgba(245, 158, 11, 0.85)",  // Kamyon (Amber)
                                "rgba(168, 85, 247, 0.85)",  // Otobüs (Mor)
                                "rgba(16, 185, 129, 0.85)"   // Motosiklet (Yeşil)
                            ],
                            borderColor: "#0F172A",
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: "right",
                                labels: {
                                    color: "#E2E8F0",
                                    font: { size: 11 },
                                    boxWidth: 12,
                                    padding: 12
                                }
                            }
                        }
                    }
                });
            }
        }

    } catch (err) {
        console.error("Grafik yükleme hatası:", err);
    }
}

// ------------------------------------------------------------------------------
// Dosya Sürükle & Bırak (Drag & Drop Video Yükleme)
// ------------------------------------------------------------------------------
function setupDropzone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const videoInput = document.getElementById("videoSourceInput");

    videoInput.addEventListener("input", () => { videoInput.dataset.userEdited = "1"; });

    dropZone.addEventListener("click", (e) => {
        if (e.target !== fileInput) fileInput.click();
    });

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

function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];

    const formData = new FormData();
    formData.append("video", file);

    const progressBar = document.getElementById("uploadProgress");
    const progressFill = document.getElementById("progressFill");
    const fileInput = document.getElementById("fileInput");
    progressBar.style.display = "block";
    progressFill.style.width = "0%";

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload", true);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            progressFill.style.width = percent + "%";
        }
    };

    xhr.onload = () => {
        progressBar.style.display = "none";
        fileInput.value = "";  // Aynı dosya tekrar seçilebilsin
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch (e) { /* boş */ }
        if (xhr.status === 200 && data.filepath) {
            const input = document.getElementById("videoSourceInput");
            input.value = data.filepath;
            input.dataset.userEdited = "1";
            alert(`✅ Video başarıyla yüklendi!\nAktif kaynak güncellendi: ${data.filepath}`);
        } else {
            alert("Yükleme başarısız oldu: " + (data.error || `Sunucu durum kodu: ${xhr.status}`));
        }
    };

    xhr.onerror = () => {
        progressBar.style.display = "none";
        fileInput.value = "";
        alert("Video yüklenirken bağlantı hatası oluştu.");
    };

    xhr.send(formData);
}

// ------------------------------------------------------------------------------
// Hızlı Tam Video Analizi & Hazır Oynatma Modu (Batch Processing)
// ------------------------------------------------------------------------------
let batchPollInterval = null;

function showBatchOverlay() {
    document.getElementById("batchProgressOverlay").style.display = "flex";
    document.getElementById("placeholderOverlay").style.display = "none";
    document.getElementById("finishedVideoPlayer").style.display = "none";
    document.getElementById("streamPlayer").style.display = "none";
    document.getElementById("btnDownloadProcessed").style.display = "none";
    setSystemStatus("TAM ANALİZ SÜRÜYOR", "#F59E0B");
}

function pollBatchStatus() {
    const overlay = document.getElementById("batchProgressOverlay");
    const pBar = document.getElementById("batchProgressBar");
    const pText = document.getElementById("batchPercent");
    const pFrames = document.getElementById("batchFrameCount");
    const finishedPlayer = document.getElementById("finishedVideoPlayer");
    const btnDownload = document.getElementById("btnDownloadProcessed");

    if (batchPollInterval) clearInterval(batchPollInterval);

    batchPollInterval = setInterval(async () => {
        try {
            const sRes = await fetch("/api/batch/status");
            const sData = await sRes.json();

            const prg = sData.progress || 0;
            pBar.style.width = prg + "%";
            pText.textContent = "%" + prg;
            pFrames.textContent = sData.total_frames
                ? `${sData.current_frame || 0} / ${sData.total_frames} Kare`
                : "Model yükleniyor...";

            if (sData.is_processing) return;

            clearInterval(batchPollInterval);
            batchPollInterval = null;

            if (sData.error) {
                overlay.style.display = "none";
                showLiveView();
                document.getElementById("placeholderOverlay").style.display = "flex";
                setSystemStatus("HATA", "#EF4444");
                alert("Analiz sırasında hata oluştu: " + sData.error);
                return;
            }

            if (sData.cancelled) {
                overlay.style.display = "none";
                showLiveView();
                document.getElementById("placeholderOverlay").style.display = "flex";
                setSystemStatus("BEKLEMEDE / DURDURULDU", "#10B981");
                loadRecords();
                loadGallery();
                return;
            }

            if (sData.completed) {
                pBar.style.width = "100%";
                pText.textContent = "%100";
                pFrames.textContent = "Tamamlandı!";

                setTimeout(() => {
                    overlay.style.display = "none";
                    // İşlenmiş videoyu oynatıcıya bağla
                    finishedPlayer.src = "/api/processed_video?t=" + Date.now();
                    finishedPlayer.style.display = "block";
                    finishedPlayer.play().catch(() => {});
                    btnDownload.style.display = "inline-flex";
                    setSystemStatus("ANALİZ TAMAMLANDI", "#10B981");

                    loadRecords();
                    loadGallery();
                    if (document.getElementById("tab-charts")?.classList.contains("active")) {
                        loadAnalyticsCharts();
                    }
                }, 800);
            }
        } catch (pollErr) {
            console.error("Durum sorgulama hatası:", pollErr);
        }
    }, 1000);
}

async function startBatchProcessing() {
    const videoPath = document.getElementById("videoSourceInput").value.trim();
    const speedLimit = parseInt(document.getElementById("speedLimitInput").value, 10);

    // Canlı akışı durdur
    if (isStreaming) {
        await stopStream();
    }

    try {
        const res = await fetch("/api/batch/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video_path: videoPath, speed_limit: speedLimit })
        });
        const data = await readJson(res);

        if (data.status === "already_running") {
            alert("Şu anda devam eden bir analiz işlemi bulunuyor. Lütfen tamamlanmasını bekleyin.");
            return;
        }
        if (!res.ok) {
            alert("Analiz başlatılamadı:\n" + (data.error || `Sunucu durum kodu: ${res.status}`));
            return;
        }

        // İlerleme arayüzünü göster
        document.getElementById("batchProgressBar").style.width = "0%";
        document.getElementById("batchPercent").textContent = "%0";
        document.getElementById("batchFrameCount").textContent = "Başlatılıyor...";
        showBatchOverlay();
        pollBatchStatus();

    } catch (err) {
        console.error("Toplu işleme hatası:", err);
        alert("Sunucuya bağlanırken bir hata oluştu.");
    }
}

async function resumeBatchIfRunning() {
    try {
        const res = await fetch("/api/batch/status");
        const data = await res.json();
        if (data.is_processing) {
            showBatchOverlay();
            pollBatchStatus();
        }
    } catch (e) {
        // yoksay
    }
}
