import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tqdm import tqdm
import joblib  # LabelEncoder nesnesini kaydetmek için


# Görselleri ve etiketleri yükle
def load_images_and_labels(data_path, ascii_path, limit=None):
    images = []
    labels = []

    # words.txt dosyasını oku
    ascii_file = os.path.join(ascii_path, "words.txt")
    with open(ascii_file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Loading images"):  # İlerleme çubuğu
        if line.startswith("#"):  # Yorumu geç
            continue

        parts = line.strip().split(" ")
        if len(parts) < 9 or parts[1] != "ok":  # Geçersiz satırları atla
            continue

        image_name = parts[0]
        label = parts[-1]

        # Görüntü dosyasının alt klasörlerdeki yolunu oluştur
        image_path = None
        for subdir, _, files in os.walk(data_path):  # Alt klasörleri gez
            if image_name + ".png" in files:
                image_path = os.path.join(subdir, image_name + ".png")
                break

        # Görüntü dosyasını bulamazsak geç
        if not image_path:
            print(f"Image not found: {image_name}.png")
            continue

        # Görüntüyü yükle ve etiketle
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue

        images.append(img)
        labels.append(label)

        # Sınırlandırma
        if limit and len(images) >= limit:
            break

    return images, labels


# Eğitim verisini yükle
DATA_PATH = "dataset/words"
ASCII_PATH = "dataset/ascii"
LIMIT = 100000  # Sadece 5000 görsel kullan (tüm veriler için None yap)
# Görselleri ve etiketleri yükle
images, labels = load_images_and_labels(DATA_PATH, ASCII_PATH, limit=None)

# Yalnızca geçerli görselleri seç
valid_images = []
valid_labels = []
for img, label in zip(images, labels):
    if img is not None:
        valid_images.append(img)
        valid_labels.append(label)
    else:
        print("Invalid image skipped.")

# Görselleri yeniden boyutlandır ve normalize et
try:
    images_resized = [cv2.resize(img, (128, 32)) for img in valid_images]
except cv2.error as e:
    print(f"Resize error: {e}")
    exit()

images_normalized = np.array(images_resized) / 255.0
images_normalized = images_normalized.reshape(-1, 32, 128, 1)

# Etiketleri encode etme
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(valid_labels)
encoded_labels = tf.keras.utils.to_categorical(encoded_labels)

# Eğitim ve doğrulama verilerine ayırma
X_train, X_val, y_train, y_val = train_test_split(images_normalized, encoded_labels, test_size=0.2, random_state=42)

# Modeli tanımlama
model = models.Sequential([
    layers.InputLayer(input_shape=(32, 128, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), verbose=1)

# LabelEncoder tanımlama ve kaydetme
joblib.dump(label_encoder, "label_encoder.pkl")  # LabelEncoder'ı kaydet

# Modeli kaydetme
model.save('handwriting_recognition_model.h5')
print("Model başarıyla kaydedildi!")
