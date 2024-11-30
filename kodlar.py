# Gerekli kütüphaneler
import pandas as pd       # Veri işleme
import numpy as np        # Matematiksel işlemler
import matplotlib.pyplot as plt  # Görselleştirme
import seaborn as sns     # İleri görselleştirme
# CSV dosyalarını yükleme
train_df = pd.read_csv("train.csv")  # Eğitim veri seti
test_df = pd.read_csv("test.csv")    # Test veri seti
submission_df = pd.read_csv("sample_submission.csv")  # Örnek çıktı dosyası


# İlk birkaç satırı görüntüleme
print(train_df.head())

# Veri seti hakkında bilgi
print(train_df.info())

# Eksik değerlerin kontrolü
print(train_df.isnull().sum())

# Veri setindeki eksik değerleri kontrol et
print(train_df.isnull().sum())  # Eğitim veri setinde eksik değerler
print(test_df.isnull().sum())   # Test veri setinde eksik değerler

# Tabloyu daha okunabilir bir şekilde göstermek için stil ekleme
train_df.head(10).style.set_table_styles(
    [{'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold')]},
     {'selector': 'td', 'props': [('border', '1px solid #ccc')]}]
)

train_df.iloc[:4]

print(f"Veri seti {train_df.shape[0]} satır ve {train_df.shape[1]} sütundan oluşuyor.")

# Sütunları kontrol edelim
print(train_df.columns)



# Eğitim verisini ayırma
X_train = train_df.drop('label', axis=1).values  # 'label' dışındaki sütunlar
y_train = train_df['label'].values  # 'label' hedef sütunu

# Eğer test etiketleri yoksa:
# Test verisi yalnızca özellikleri içerir
X_test = test_df.values  # Eğer 'label' sütunu yoksa doğrudan test_df kullanılır

# Reshape işlemi: Görüntü formatı (28x28x1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # Eğitim verisi
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)  # Test verisi

# Normalizasyon (0-255 değerlerini 0-1 arasına çekme)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# CNN Modeli oluşturma
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # İlk evrişim katmanı
    layers.MaxPooling2D((2, 2)),  # MaxPooling ile boyut küçültme
    layers.Conv2D(64, (3, 3), activation='relu'),  # İkinci evrişim katmanı
    layers.MaxPooling2D((2, 2)),  # MaxPooling
    layers.Conv2D(64, (3, 3), activation='relu'),  # Üçüncü evrişim katmanı
    layers.Flatten(),  # Düzleştirme katmanı (full connected katmanlar için)
    layers.Dense(64, activation='relu'),  # Tam bağlantılı katman
    layers.Dense(10, activation='softmax')  # 10 sınıf (rakamlar 0-9)
])

# Modeli derleme (optimizer ve loss fonksiyonu seçiyoruz)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modelin özetini yazdırma
model.summary()


# Modeli eğitme (10 epoch boyunca)
model.fit(train_data, train_labels, epochs=10, batch_size=64)

# Modeli test verisiyle değerlendirme
test_loss, test_acc = model.evaluate(test_data, test_labels)

print(f'Test accuracy: {test_acc}')

# Test verisinden bir örnek seçip tahmin yapalım
predictions = model.predict(test_data)

# İlk test örneğini ve modelin tahminini gösterelim
print(f'Tahmin: {np.argmax(predictions[0])}')
print(f'Gerçek Değer: {test_labels[0]}')

# İlk test örneğini görselleştirelim
import matplotlib.pyplot as plt

plt.imshow(test_data[0].reshape(28, 28), cmap='gray')
plt.show()

# CNN modelinin eğitimini başlatma
history = model.fit(
    X_train,              # Eğitim verisi
    y_train,              # Eğitim etiketleri
    epochs=10,            # Eğitim epok sayısı
    validation_data=(X_train , y_train),  # Doğrulama verisi ve etiketleri
    batch_size=32         # Mini-batch boyutu
)

import matplotlib.pyplot as plt

# Eğitim kaybı ve doğruluk grafiğini çizmek
plt.figure(figsize=(14, 6))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', color='blue')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', color='orange')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok Sayısı')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı', color='red')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='green')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok Sayısı')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 1'den 9'a kadar olan sayılar için tahmin yapalım
fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # 3x3 grid oluşturuyoruz
axes = axes.flatten()  # Axes'i düzleştiriyoruz ki her bir grafik için index kullanabilelim

for i in range(1, 10):
    # Test verisinde i sayısına karşılık gelen tüm örnekleri seçme
    indices = np.where(test_labels == i)[0]

    if len(indices) > 0:
        # İlk örneği seçelim
        sample_index = indices[0]
        sample_image = test_data[sample_index]

        # Tahmin yapalım
        sample_prediction = model.predict(sample_image.reshape(1, 28, 28, 1))  # model için uygun boyut

        # Sonuçları yazdıralım
        predicted_class = np.argmax(sample_prediction)
        actual_class = test_labels[sample_index]

        # Görseli ve başlıkları yan yana göstermek için subplots kullanalım
        axes[i-1].imshow(sample_image.reshape(28, 28), cmap='gray')
        axes[i-1].set_title(f"Tahmin: {predicted_class}\nGerçek: {actual_class}")
        axes[i-1].axis('off')  # Eksenleri kaldırıyoruz

# Alt başlık ekleyelim
fig.suptitle("Tahminler ve Gerçek Değerler (1-9)", fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Başlık için biraz boşluk bırakalım
plt.show()
