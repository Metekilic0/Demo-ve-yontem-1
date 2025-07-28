# Gerekli kütüphaneleri projemize dahil ediyoruz
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern ### YENİ ###
from tqdm import tqdm
import joblib

print("Süper Zengin Öznitelik (HOG + Renk + LBP + Augmentation) Çıkarma İşlemi Başlatılıyor...")

# --- 1. Proje Ayarları ve Veri Yolları ---
BASE_PATH = "C:/Users/mkasl/Desktop/donem5/yap470/Progress rapor3/dataset4"
TRAIN_PATH = os.path.join(BASE_PATH, 'train')

IMAGE_SIZE = (128, 128)
# HOG parametreleri
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9
# Renk Histogramı için 'bin' sayısı
BINS = 8
### YENİ ###: LBP PARAMETRELERİ
LBP_POINTS = 24 # Komşuluktaki nokta sayısı
LBP_RADIUS = 8  # Noktaların merkezden uzaklığı (yarıçap)
### BİTTİ ###


# --- 2. Yardımcı Fonksiyonlar ---
def augment_color_image(bgr_image):
    augmented_images = []
    augmented_images.append(bgr_image)
    flipped_image = cv2.flip(bgr_image, 1)
    augmented_images.append(flipped_image)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_brighter = cv2.add(v, 40)
    brighter_hsv = cv2.merge([h, s, v_brighter])
    brighter_image = cv2.cvtColor(brighter_hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(brighter_image)
    return augmented_images

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
            original_color_image = cv2.imread(image_path)
            if original_color_image is None:
                print(f"Uyarı: {image_path} okunamadı, atlanıyor.")
                continue

            images_to_process = augment_color_image(original_color_image)
            
            for color_image in images_to_process:
                color_image = cv2.resize(color_image, IMAGE_SIZE)
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # 1. HOG Özniteliklerini Çıkar
                hog_features = hog(gray_image, orientations=ORIENTATIONS,
                                   pixels_per_cell=PIXELS_PER_CELL,
                                   cells_per_block=CELLS_PER_BLOCK,
                                   block_norm='L2-Hys')

                # 2. Renk Histogramı Özniteliklerini Çıkar
                color_features = extract_color_histogram(color_image, bins=(BINS, BINS, BINS))
                
                ### YENİ ###: LBP (DOKU) ÖZNİTELİKLERİNİ ÇIKAR
                lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method="uniform")
                (lbp_hist, _) = np.histogram(lbp.ravel(),
                                             bins=np.arange(0, LBP_POINTS + 3),
                                             range=(0, LBP_POINTS + 2))
                lbp_hist = lbp_hist.astype("float")
                lbp_hist /= (lbp_hist.sum() + 1e-6) # Normalize et
                ### BİTTİ ###
                
                ### DEĞİŞTİRİLDİ ###: Üç özellik vektörünü birleştir
                combined_features = np.concatenate([hog_features, color_features, lbp_hist])
                ### BİTTİ ###

                features.append(combined_features)
                labels.append(class_name)
                
        except Exception as e:
            print(f"Hata: {image_path} işlenemedi. Detay: {e}")

print("\nSüper zengin öznitelik çıkarma tamamlandı!")

# --- 4. Çıkarılan Zengin Özellikleri Kaydetme ---
features = np.array(features, dtype=np.float32)
labels = np.array(labels)

print(f"Toplam {len(features)} adet görüntü için öznitelik vektörü oluşturuldu.")
print(f"Yeni birleşik öznitelik vektörlerinin boyutu: {features.shape[1]}")

data_to_save = {'features': features, 'labels': labels}

### DEĞİŞTİRİLDİ ###: Yeni ve daha açıklayıcı dosya adı
output_filename = 'hog_color_lbp_augmented_4class.pkl'
joblib.dump(data_to_save, output_filename)

print(f"\nHarika! Tüm birleşik öznitelikler '{output_filename}' dosyasına kaydedildi.")