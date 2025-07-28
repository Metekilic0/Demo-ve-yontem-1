import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm
import joblib

print("Zengin ve Artırılmış Öznitelik (HOG + Renk + Augmentation) Çıkarma İşlemi Başlatılıyor...")

# --- 1. Proje Ayarları ve Veri Yolları ---
BASE_PATH = "C:/Users/mkasl/Desktop/donem5/yap470/Progress rapor3/dataset4"
TRAIN_PATH = os.path.join(BASE_PATH, 'train')

IMAGE_SIZE = (128, 128)
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9
BINS = 8

# --- 2. Yardımcı Fonksiyonlar ---

### YENİ ###: VERİ ARTIRMA FONKSİYONU (RENKLİ GÖRÜNTÜLER İÇİN)
def augment_color_image(bgr_image):
    """
    BGR formatındaki bir renkli görüntüye veri artırma teknikleri uygular.
    Orijinal, yatay çevrilmiş ve parlaklığı değiştirilmiş görüntülerin bir listesini döndürür.
    """
    augmented_images = []
    
    # 1. Orijinal Görüntü
    augmented_images.append(bgr_image)
    
    # 2. Yatay Çevrilmiş Görüntü
    flipped_image = cv2.flip(bgr_image, 1)
    augmented_images.append(flipped_image)
    
    # 3. Parlaklığı Artırılmış Görüntü
    # Görüntüyü HSV formatına çevir, V (Value/Parlaklık) kanalını artır
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_brighter = cv2.add(v, 40) # Değerleri 40 birim artır
    brighter_hsv = cv2.merge([h, s, v_brighter])
    brighter_image = cv2.cvtColor(brighter_hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(brighter_image)
    
    return augmented_images
### YENİ SONU ###

def extract_color_histogram(image, bins=(BINS, BINS, BINS)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- 3. Öznitelik Çıkarma ve Birleştirme Döngüsü ---
features = []
labels = []

for class_name in tqdm(os.listdir(TRAIN_PATH), desc="Sınıflar işleniyor"):
    class_path = os.path.join(TRAIN_PATH, class_name)
    if not os.path.isdir(class_path): continue

    for image_name in tqdm(os.listdir(class_path), desc=f"'{class_name}' sınıfı", leave=False):
        image_path = os.path.join(class_path, image_name)
        try:
            # Resmi RENKLİ olarak oku
            original_color_image = cv2.imread(image_path)
            if original_color_image is None:
                print(f"Uyarı: {image_path} okunamadı, atlanıyor.")
                continue

            ### DEĞİŞTİRİLDİ ###: Veri artırmayı uygula
            # Artık orijinal görüntü yerine artırılmış görüntü listesini işleyeceğiz
            images_to_process = augment_color_image(original_color_image)
            
            # Her bir artırılmış versiyon için öznitelik çıkar
            for color_image in images_to_process:
                color_image = cv2.resize(color_image, IMAGE_SIZE)
                
                # Gri tonlamalı versiyonu HOG için oluştur
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # 1. HOG Özniteliklerini Çıkar
                hog_features = hog(gray_image, orientations=ORIENTATIONS,
                                   pixels_per_cell=PIXELS_PER_CELL,
                                   cells_per_block=CELLS_PER_BLOCK,
                                   block_norm='L2-Hys')

                # 2. Renk Histogramı Özniteliklerini Çıkar
                color_features = extract_color_histogram(color_image, bins=(BINS, BINS, BINS))
                
                # 3. İki özellik vektörünü birleştirerek "süper-vektörü" oluştur
                combined_features = np.concatenate([hog_features, color_features])

                features.append(combined_features)
                labels.append(class_name)
                
        except Exception as e:
            print(f"Hata: {image_path} işlenemedi. Detay: {e}")

print("\nZengin ve artırılmış öznitelik çıkarma tamamlandı!")

# --- 4. Çıkarılan Zengin Özellikleri Kaydetme ---
features = np.array(features, dtype=np.float32)
labels = np.array(labels)

print(f"Toplam {len(features)} adet görüntü için öznitelik vektörü oluşturuldu.")
print(f"Yeni birleşik öznitelik vektörlerinin boyutu: {features.shape[1]}")

data_to_save = {'features': features, 'labels': labels}

### DEĞİŞTİRİLDİ ###: Yeni dosya adı
output_filename = 'hog_color_augmented_features_4class.pkl'
joblib.dump(data_to_save, output_filename)

print(f"\nHarika! Tüm birleşik öznitelikler '{output_filename}' dosyasına kaydedildi.")
print("Şimdi bu yeni veriyle PCA ve model eğitimi yapmaya hazırsın!")