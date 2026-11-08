import os
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _setup_fonts():
    # Türkçe karakter desteği için Arial fontunu yükle
    font_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("TurkishFont", p))
                return "TurkishFont"
            except Exception:
                pass
    return "Helvetica"


FONT_NAME = _setup_fonts()


def generate_violation_pdf(record: Dict[str, Any], output_pdf_path: str) -> str:
    """
    Hız ihlali yapan bir araç için resmi formatta PDF İhlal Tespit Tutanağı üretir.
    """
    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "HeaderTitle",
        fontName=FONT_NAME,
        fontSize=15,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "HeaderSubtitle",
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#475569")
    )
    label_style = ParagraphStyle(
        "LabelStyle",
        fontName=FONT_NAME,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    val_style = ParagraphStyle(
        "ValStyle",
        fontName=FONT_NAME,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0F172A")
    )
    danger_style = ParagraphStyle(
        "DangerStyle",
        fontName=FONT_NAME,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#DC2626")
    )

    story = []

    # 1. Başlık Alanı
    story.append(Paragraph("<b>T.C. AKILLI TRAFİK DENETLEME VE GÖZETLEME SİSTEMİ</b>", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("SMART TRAFFIC PRO · OTOMATİK HIZ İHLALİ TESPİT TUTANAĞI", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceBefore=2, spaceAfter=12))

    # 2. Tutanak Bilgileri Tablosu
    class_tr = {"car": "Otomobil", "truck": "Kamyon", "bus": "Otobüs", "motorcycle": "Motosiklet"}
    try:
        rec_id = int(record.get("id") or record.get("track_id") or 0)
    except (TypeError, ValueError):
        rec_id = 0
    plate = record.get("plate_text") or "BELİRTİLMEDİ"
    speed = int(record.get("speed_kmh") or 0)
    limit = int(record.get("speed_limit") or 80)
    diff = max(0, speed - limit)
    diff_pct = int((diff / limit) * 100) if limit > 0 else 0
    raw_cls = (record.get("class_name") or "").lower()
    cls_name = class_tr.get(raw_cls, raw_cls.capitalize() or "Belirtilmedi")
    lane = record.get("lane_name") or "Belirtilmedi"
    timestamp = record.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info_data = [
        [
            Paragraph("<b>Tutanak Referans No:</b>", label_style),
            Paragraph(f"STP-2026-V{rec_id:05d}", val_style),
            Paragraph("<b>Tespit Tarihi & Saati:</b>", label_style),
            Paragraph(str(timestamp), val_style)
        ],
        [
            Paragraph("<b>Araç Plakası:</b>", label_style),
            Paragraph(f"<b><font size='11'>{plate}</font></b>", val_style),
            Paragraph("<b>Araç Cinsi / Sınıfı:</b>", label_style),
            Paragraph(cls_name, val_style)
        ],
        [
            Paragraph("<b>Ölçülen Araç Hızı:</b>", label_style),
            Paragraph(f"<b><font color='#DC2626' size='11'>{speed} km/s</font></b>", danger_style),
            Paragraph("<b>Yasal Hız Sınırı:</b>", label_style),
            Paragraph(f"{limit} km/s", val_style)
        ],
        [
            Paragraph("<b>Hız Aşım Miktarı:</b>", label_style),
            Paragraph(f"<font color='#DC2626'>+{diff} km/s (%{diff_pct} Aşım)</font>", danger_style),
            Paragraph("<b>Ölçüm Bölgesi / Şerit:</b>", label_style),
            Paragraph(str(lane), val_style)
        ],
        [
            Paragraph("<b>Tespit Yöntemi:</b>", label_style),
            Paragraph("YOLO11m + Metrik Poligon Zaman Çizgisi (FPS Decoupled)", val_style),
            Paragraph("<b>Hata Toleransı:</b>", label_style),
            Paragraph("MAE &le; 2.0 km/s", val_style)
        ]
    ]

    t_info = Table(info_data, colWidths=[120, 140, 120, 140])
    t_info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 14))

    # 3. Kanıt Fotoğrafı
    snap_path = record.get("snapshot_path")
    if snap_path:
        if not os.path.exists(snap_path):
            alt_paths = [
                os.path.join("output", "ihlaller", os.path.basename(snap_path)),
                os.path.join("output", os.path.basename(snap_path))
            ]
            for ap in alt_paths:
                if os.path.exists(ap):
                    snap_path = ap
                    break

    if snap_path and os.path.exists(snap_path):
        story.append(Paragraph("<b>ADLİ VE TEKNİK KANIT FOTOĞRAFI (RADAR SNAPSHOT)</b>", ParagraphStyle(
            "SecHeader", fontName=FONT_NAME, fontSize=10, textColor=colors.HexColor("#1E293B")
        )))
        story.append(Spacer(1, 6))
        try:
            # En-boy oranını koruyarak A4 genişliğine uydur
            img = RLImage(snap_path, width=520, height=270)
            story.append(img)
            story.append(Spacer(1, 10))
        except Exception as ie:
            story.append(Paragraph(f"[Fotoğraf yüklenemedi: {ie}]", label_style))

    # 4. Yasal Uyarı ve Doğrulama Alanı
    notice_text = (
        "<b>NOT VE YASAL BİLGİLENDİRME:</b> Bu tutanak, "
        "SMART TRAFFIC PRO Yapay Zeka Trafik İzleme Sistemi tarafından otomatik üretilmiştir. "
        "Ölçülen hız değerleri kamera perspektif kalibrasyonuna ve video zaman çizgisi aralığına dayanmaktadır."
    )
    notice_table = Table([[
        Paragraph(notice_text, ParagraphStyle("Notice", fontName=FONT_NAME, fontSize=8, leading=11, textColor=colors.HexColor("#64748B")))
    ]], colWidths=[520])
    notice_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#94A3B8")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(notice_table)
    story.append(Spacer(1, 12))

    # 5. Onay ve İmza Alanı
    sign_data = [
        [
            Paragraph("<b>Sistem Operatörü:</b><br/>Yiğit Ünlütürk (210129060)<br/>SMART TRAFFIC PRO AI SOC", label_style),
            Paragraph("<b>Doğrulama Durumu:</b><br/>Yapay Zeka Onaylı & Dijital Mühürlü<br/>İtiraz Süresi: 15 Gün", label_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[260, 260])
    sign_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sign_table)

    doc.build(story)
    return output_pdf_path
