import cv2
import numpy as np

# 1. Görüntüyü Oku ve Boyutlarını Al
image = cv2.imread('nesneler1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
yukseklik, genislik = image.shape[:2]

# 2. Bulanıklaştırma (Gürültü engelleme)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. EŞİKLEME (Siyah-Beyaz Maske Oluşturma)
# Otsu algoritması ile ışığı otomatik ayarlar
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. MORFOLOJİK İŞLEMLER (Temizlik)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# 5. NESNE BULMA
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

nesne_sayisi = 0

for cnt in contours:
    alan = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # --- ÜÇLÜ FİLTRELEME SİSTEMİ ---
    
    # A) ALAN FİLTRESİ: Çok küçük tozları ve kağıdın tamamını eler.
    if 1000 < alan < 150000:
        
        # B) KALINLIK FİLTRESİ: İnce gölge çizgilerini eler.
        if w > 35 and h > 35:
            
            # C) KENAR (ÖLÜ BÖLGE) FİLTRESİ: 
            # Nesne sağ kenara çok yakınsa (gölge bölgesi) onu sayma.
            if (x + w) < (genislik - 150):
                nesne_sayisi += 1
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Sonucu ekrana yazdır
cv2.putText(image, f'Bulunan Nesne: {nesne_sayisi}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 6. GÖRÜNTÜLEME
cv2.imshow('1- Esikleme Maskesi', thresh)
cv2.imshow('2- Temizlenmis Maske', closing)
cv2.imshow('3- Final Sonuc', image)

cv2.waitKey(0)
cv2.destroyAllWindows()