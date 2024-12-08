import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd  # Excel dosyasını okumak için
from tqdm import tqdm

# Modeli ve LabelEncoder'ı yükleme
model = tf.keras.models.load_model('handwriting_recognition_model.h5')
label_encoder = joblib.load("label_encoder.pkl")

# Excel dosyasını oku
def load_labels_from_excel(file_path):
    try:
        data = pd.read_csv(file_path, header=None, names=["Path", "Label"])
        return data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# Test görsellerini yükleme fonksiyonu
def load_test_images_from_excel(data_frame):
    images = []
    labels = []
    file_names = []

    for _, row in data_frame.iterrows():
        image_path = row["Path"]
        label = row["Label"]

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Görüntü yüklenebiliyor mu kontrol et
                images.append(img)
                labels.append(label)
                file_names.append(image_path)
            else:
                print(f"Could not read image: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return images, labels, file_names

# Excel dosyası yolunu ayarla
EXCEL_FILE_PATH = "preprocessed.csv"  # Burayı uygun dosya yoluna güncelle

# Excel'den veriyi yükle
data = load_labels_from_excel(EXCEL_FILE_PATH)
if data is None:
    raise ValueError("Excel file could not be loaded. Please check the file format and path.")

# Görselleri ve etiketleri yükle
test_images, real_labels, file_names = load_test_images_from_excel(data)

# Görselleri yeniden boyutlandırma
test_images_resized = []
for img in test_images:
    try:
        resized_img = cv2.resize(img, (128, 32))
        test_images_resized.append(resized_img)
    except Exception as e:
        print(f"Error resizing image: {e}")

if len(test_images_resized) == 0:
    raise ValueError("No valid test images found. Please check the input folder and image format.")

# Görselleri normalize etme
test_images_normalized = np.array(test_images_resized) / 255.0
test_images_normalized = test_images_normalized.reshape(-1, 32, 128, 1)

# Tahminler
if test_images_normalized.size > 0:
    predictions = model.predict(test_images_normalized)
    predicted_labels = np.argmax(predictions, axis=1)
    decoded_labels = label_encoder.inverse_transform(predicted_labels)

    # Eşleşen ve eşleşmeyen sonuçları sayma
    match_count = 0
    mismatch_count = 0

    for real_label, predicted_label in zip(real_labels, decoded_labels):
        if real_label == predicted_label:
            match_count += 1
        else:
            mismatch_count += 1

    # Tahminleri ve gerçek etiketleri görselleştir
    '''
    for i, (img, prediction, real_label, file_name) in enumerate(zip(test_images, decoded_labels, real_labels, file_names)):
        plt.imshow(img, cmap='gray')
        plt.title(f"File: {file_name}\nReal: {real_label}\nPrediction: {prediction}")
        plt.axis('off')
        plt.show()
    '''

    # Eşleşen ve eşleşmeyen sonuçları yazdır
    print(f"Total Matches: {match_count}")
    print(f"Total Mismatches: {mismatch_count}")
else:
    print("No images to predict. Check your test data.")
