import argparse
import os
import sys
from tqdm import tqdm

from src.pipeline import TrafficPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eşzamanlı Araç Hız Tespiti ve Plaka Tanıma Sistemi (Traffic Speed & ALPR System)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="video.mp4",
        help="Girdi video dosyası yolu veya kamera indeksi (örn: 0, video.mp4, trafik.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/output_combined.mp4",
        help="İşlenmiş çıktı video dosya yolu"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default_config.yaml",
        help="Konfigürasyon YAML dosya yolu"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="İşlem sırasında OpenCV penceresinde canlı göster"
    )
    parser.add_argument(
        "--speed-limit",
        type=int,
        default=None,
        help="Hız limiti (km/h) - Config dosyasındaki değeri geçersiz kılar"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Hesaplama cihazı ('cpu' veya '0', 'cuda')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 EŞZAMANLI ARAÇ HIZ VE PLAKA TESPİT SİSTEMİ")
    print("=" * 70)

    if not os.path.exists(args.config):
        print(f"❌ Hata: Konfigürasyon dosyası bulunamadı: {args.config}")
        sys.exit(1)

    print(f"⚙️ Konfigürasyon: {args.config}")
    print(f"📹 Girdi Video: {args.input}")
    print(f"💾 Çıktı Video: {args.output}")

    pipeline = TrafficPipeline(config=args.config)

    if args.speed_limit is not None:
        pipeline.speed_limit = args.speed_limit
        pipeline.speed_estimator.speed_limit = args.speed_limit
        pipeline.violation_manager.speed_limit = args.speed_limit
        pipeline.visualizer.speed_limit = args.speed_limit

    if args.device is not None:
        pipeline.detector.device = args.device

    input_source = int(args.input) if args.input.isdigit() else args.input

    if isinstance(input_source, str) and not os.path.exists(input_source):
        print(f"\n⚠️ UYARI: Girdi videosu '{input_source}' bulunamadı.")
        print("Mevcut örnek videolardan birini seçebilir veya parametre olarak verebilirsiniz:")
        print("Örnek çalıştırma: python main.py --input video.mp4 --show")
        return

    print("\n⏳ Video işleniyor, lütfen bekleyiniz...\n")
    pbar = None

    def on_progress(current, total, summary):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="İşlenen Kareler", unit="frame")
        pbar.update(1)
        pbar.set_postfix({
            "Arac": summary.get("total_vehicles", 0),
            "Plaka": summary.get("plate_recognized_count", 0),
            "Ihlal": summary.get("violations_count", 0)
        })

    try:
        summary = pipeline.process_video(
            video_path=input_source,
            output_path=args.output,
            show_window=args.show,
            progress_callback=on_progress
        )
    finally:
        if pbar is not None:
            pbar.close()

    print("\n" + "=" * 70)
    print("📊 İŞLEM TAMAMLANDI - SONUÇ ÖZETİ")
    print("=" * 70)
    print(f"🚗 Toplam Tespit Edilen Araç : {summary.get('total_vehicles', 0)}")
    print(f"⚡ Hızı Ölçülen Araç Sayısı   : {summary.get('speed_measured_count', 0)}")
    print(f"🪪 Plakası Okunan Araç       : {summary.get('plate_recognized_count', 0)}")
    print(f"🚨 Hız İhlali Yapan Araç     : {summary.get('violations_count', 0)}")
    print(f"📈 Ortalama Hız               : {summary.get('average_speed', 0)} km/h")
    print(f"🏁 Kaydedilen Maksimum Hız   : {summary.get('max_speed', 0)} km/h")
    print(f"📄 CSV Kayıt Dosyası         : {summary.get('csv_path', '-')}")
    if "output_video" in summary:
        print(f"🎥 Çıktı Videosu             : {summary.get('output_video', '-')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
