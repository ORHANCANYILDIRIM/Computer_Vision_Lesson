import cv2
import csv
import requests
from ultralytics import YOLO
import os

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")

# İndirilen resimleri kaydetmek için klasör oluştur
download_folder = 'downloaded_images'
os.makedirs(download_folder, exist_ok=True)


# Resmi URL'den indirip belirli bir klasöre kaydetme fonksiyonu
def download_image(url, folder=download_folder):
    response = requests.get(url)
    if response.status_code == 200:
        image_name = os.path.join(folder, url.split('/')[-1].split('?')[0])
        with open(image_name, 'wb') as file:
            file.write(response.content)
        return image_name
    else:
        print(f"Resim indirilemedi: {url}")
        return None


# Verilen URL'lerde insan tespiti yaparak sonuçları günceller
def process_images_from_csv(csv_path='results.csv'):
    rows = []

    # CSV dosyasını okuyup URL'leri al
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 1:  # Eğer sadece URL varsa
                url = row[0]

                # Resmi indir
                image_path = download_image(url)
                if not image_path:
                    continue

                # Modeli çalıştır ve insan tespiti yap
                results = model.predict(image_path)
                is_human = any(int(box[5]) == 0 for result in results for box in result.boxes.data)

                # Sonucu satıra ekle
                row.append("Human" if is_human else "Not Human")

            # Daha önce analiz edilmiş satırlar varsa onları da ekle
            rows.append(row)

    # Sonuçları tekrar aynı CSV dosyasına yaz
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Analiz tamamlandı ve sonuçlar results.csv dosyasına kaydedildi.")


# İşlem başlat
process_images_from_csv()
