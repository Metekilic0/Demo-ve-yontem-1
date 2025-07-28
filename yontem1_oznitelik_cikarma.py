# Gerekli kütüphaneleri projemize dahil ediyoruz
import os
import cv2 # Görüntü işleme için OpenCV kütüphanesi
import numpy as np
from skimage.feature import hog # HOG özelliklerini çıkarmak için scikit-image
from tqdm import tqdm # İşlem ilerlemesini gösteren şık bir ilerleme çubuğu için
import joblib # Büyük veri yapılarını verimli bir şekilde kaydetmek için

# --- 1. Proje Ayarları ve Veri Yolları ---
# Bu bölümde, işlem yapacağımız klasörleri ve ayarları tanımlıyoruz.

# Düzenlenmiş ve 4 sınıflı veri setimizin ana klasör yolu
BASE_PATH = "C:/Users/mkasl/Desktop/donem5/yap470/Progress rapor3/dataset4"
# Eğitim verilerinin bulunduğu klasör
TRAIN_PATH = os.path.join(BASE_PATH, 'train')

# HOG özelliklerini çıkarırken resimleri standartlaştıracağımız boyut
# Bu boyut, özellik vektörünün tutarlılığı için önemlidir.
IMAGE_SIZE = (128, 128)
# HOG parametreleri: Bu değerler, HOG'un ne kadar detaylı özellik çıkaracağını belirler.
# Bunlar da birer hiper-parametredir ve değiştirilebilir.
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

print("HOG Öznitelik Çıkarma İşlemi Başlatılıyor...")
print(f"Veri Kaynağı: {TRAIN_PATH}")

# --- 2. Öznitelik Çıkarma ve Etiketleme Döngüsü ---
# Bu bölümde, tüm eğitim verilerini tek tek gezip HOG özelliklerini çıkaracağız.

# Çıkardığımız özellikleri (sayısal vektörler) saklamak için boş bir liste
features = []
# Bu özelliklerin hangi sınıfa ait olduğunu saklamak için boş bir liste
labels = []

# Eğitim klasöründeki her bir sınıf klasörünü ('airplane', 'bird'...) gez
for class_name in tqdm(os.listdir(TRAIN_PATH), desc="Sınıflar işleniyor"):
    class_path = os.path.join(TRAIN_PATH, class_name)
    
    # Eğer bu bir klasör değilse, atla
    if not os.path.isdir(class_path):
        continue

    # O sınıf klasörünün içindeki her bir resim dosyasını gez
    for image_name in tqdm(os.listdir(class_path), desc=f"'{class_name}' sınıfı", leave=False):
        image_path = os.path.join(class_path, image_name)

        try:
            # Resmi gri tonlamalı olarak oku. HOG, renk bilgisine ihtiyaç duymaz.
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Resmi standart boyuta getir
            image = cv2.resize(image, IMAGE_SIZE)
            
            # HOG özniteliklerini çıkar
            # Bu fonksiyon, resmi analiz ederek onu temsil eden uzun bir sayı dizisi döndürür.
            hog_features = hog(image, orientations=ORIENTATIONS,
                               pixels_per_cell=PIXELS_PER_CELL,
                               cells_per_block=CELLS_PER_BLOCK,
                               block_norm='L2-Hys')
            
            # Çıkan özellikleri ve resmin etiketini (klasör adını) listelerimize ekle
            features.append(hog_features)
            labels.append(class_name)
        except Exception as e:
            # Eğer bir resim okunamazsa veya bozuksa, hatayı yazdır ve devam et.
            print(f"Hata: {image_path} işlenemedi. Detay: {e}")

print("\nÖznitelik çıkarma işlemi başarıyla tamamlandı!")

# --- 3. Çıkarılan Özellikleri Kalıcı Olarak Kaydetme ---
# Bu uzun işlemi tekrar tekrar yapmamak için, elde ettiğimiz tüm veriyi
# tek bir dosyaya kaydediyoruz. Sonraki adımlarda bu dosyayı okuyarak devam edeceğiz.

# Python listelerini, makine öğrenmesi modellerinin daha kolay işleyebileceği
# Numpy dizilerine dönüştürüyoruz.
features = np.array(features)
labels = np.array(labels)

print(f"Toplam {len(features)} adet görüntü için öznitelik vektörü oluşturuldu.")
print(f"Öznitelik vektörlerinin boyutu (bir resim için): {features.shape[1]}")

# Veriyi ve etiketleri bir sözlük yapısında birleştiriyoruz
data_to_save = {
    'features': features,
    'labels': labels
}

# joblib.dump ile veriyi sıkıştırılmış ve hızlı okunabilir bir formatta diske yazıyoruz.
output_filename = 'hog_features_4class.pkl'
joblib.dump(data_to_save, output_filename)

print(f"\nHarika! Tüm öznitelikler ve etiketler '{output_filename}' dosyasına kaydedildi.")
print("Artık Makine Öğrenmesi modellerini eğitmeye hazırsın!")