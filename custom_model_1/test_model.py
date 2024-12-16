import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd  # To read the Excel file
from tqdm import tqdm  # For progress tracking
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the pre-trained model and LabelEncoder
model = tf.keras.models.load_model('handwriting_recognition_model.h5')
label_encoder = joblib.load("label_encoder.pkl")

# Function to read the Excel file containing image paths and labels
def load_labels_from_excel(file_path):
    try:
        data = pd.read_csv(file_path, header=None, names=["Path", "Label"])
        return data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# Function to load test images and labels based on the DataFrame
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
            if img is not None:
                images.append(img)
                labels.append(label)
                file_names.append(image_path)
            else:
                print(f"Could not read image: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return images, labels, file_names

# Path to the Excel file containing test image paths and labels
EXCEL_FILE_PATH = "updated_test_values.csv"

# Load data from the Excel file
data = load_labels_from_excel(EXCEL_FILE_PATH)
if data is None:
    raise ValueError("Excel file could not be loaded. Please check the file format and path.")

# Load images and labels from the DataFrame
test_images, real_labels, file_names = load_test_images_from_excel(data)

# Resize images to the input shape required by the model
test_images_resized = []
for img in test_images:
    try:
        resized_img = cv2.resize(img, (128, 32))
        test_images_resized.append(resized_img)
    except Exception as e:
        print(f"Error resizing image: {e}")

if len(test_images_resized) == 0:
    raise ValueError("No valid test images found. Please check the input folder and image format.")

# Normalize image pixel values to the range [0, 1]
test_images_normalized = np.array(test_images_resized) / 255.0
test_images_normalized = test_images_normalized.reshape(-1, 32, 128, 1)

# Perform predictions if there are images to predict
if test_images_normalized.size > 0:
    predictions = model.predict(test_images_normalized)
    predicted_labels = np.argmax(predictions, axis=1)
    decoded_labels = label_encoder.inverse_transform(predicted_labels)

    # Count matches and mismatches
    match_count = 0
    mismatch_count = 0
    for real_label, predicted_label in zip(real_labels, decoded_labels):
        if real_label == predicted_label:
            match_count += 1
        else:
            mismatch_count += 1

    # Prepare data for graph plotting
    total_processed = len(real_labels)
    true_match = match_count
    batch_numbers = list(range(1, len(real_labels) + 1))
    batch_cer = []  # Placeholder for CER per batch
    batch_wer = []  # Placeholder for WER per batch

    # Calculate CER and WER for each batch
    def cer(true_texts, pred_texts):
        """Calculate Character Error Rate (CER)."""
        total_chars = sum(len(t) for t in true_texts)
        total_errors = sum(
            sum(1 for a, b in zip(t, p) if a != b) + abs(len(t) - len(p))
            for t, p in zip(true_texts, pred_texts)
        )
        return total_errors / total_chars if total_chars > 0 else 0

    def wer(true_texts, pred_texts):
        """Calculate Word Error Rate (WER)."""
        total_words = sum(len(t.split()) for t in true_texts)
        total_errors = sum(
            sum(1 for a, b in zip(t.split(), p.split()) if a != b) + abs(len(t.split()) - len(p.split()))
            for t, p in zip(true_texts, pred_texts)
        )
        return total_errors / total_words if total_words > 0 else 0

    batch_size = 16
    for i in range(0, len(real_labels), batch_size):
        batch_real = real_labels[i:i + batch_size]
        batch_pred = decoded_labels[i:i + batch_size]
        batch_cer.append(cer(batch_real, batch_pred))
        batch_wer.append(wer(batch_real, batch_pred))

    # Plot test results including violin plot
    plt.figure(figsize=(15, 10))

    # Accuracy Distribution
    plt.subplot(2, 2, 1)
    correct_preds = true_match
    incorrect_preds = total_processed - true_match
    plt.bar(['Correct', 'Incorrect'], [correct_preds, incorrect_preds], color=['green', 'red'])
    plt.title("Prediction Accuracy Distribution")
    plt.ylabel("Count")

    # Error Rates Over Batches
    plt.subplot(2, 2, 2)
    plt.plot(batch_numbers[:len(batch_cer)], batch_cer, label="CER", marker='o')
    plt.plot(batch_numbers[:len(batch_wer)], batch_wer, label="WER", marker='x')
    plt.title("Error Rates over Batches")
    plt.xlabel("Batch Number")
    plt.ylabel("Error Rate")
    plt.legend()

    # Confusion Matrix
    plt.subplot(2, 2, 3)
    y_true = [1 if pred.strip() == true.strip() else 0 for pred, true in zip(decoded_labels, real_labels)]
    y_pred = [1] * len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
    disp.plot(ax=plt.gca(), cmap=plt.cm.Blues)
    plt.title("Prediction Accuracy Confusion Matrix")

    # Violin Plot for Batch Accuracies
    plt.subplot(2, 2, 4)
    batch_accuracies = []
    for i in range(0, len(decoded_labels), batch_size):
        batch_pred = decoded_labels[i:i + batch_size]
        batch_true = real_labels[i:i + batch_size]
        batch_acc = sum(1 for p, t in zip(batch_pred, batch_true) if p.strip() == str(t).strip()) / len(batch_pred)
        batch_accuracies.append(batch_acc)

    plt.violinplot(batch_accuracies, positions=[1])
    plt.title("Distribution of Batch Accuracies")
    plt.ylabel("Accuracy")
    plt.xticks([1], ['Batches'])
    plt.ylim(0, 1)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print Statistics
    print("\nTest Results:")
    print(f"Total images processed: {total_processed}")
    print(f"Correct predictions: {true_match}")
    print(f"Accuracy: {(true_match / total_processed) * 100:.2f}%")
    print(f"Character Error Rate (CER): {cer(real_labels, decoded_labels):.4f}")
    print(f"Word Error Rate (WER): {wer(real_labels, decoded_labels):.4f}")
else:
    print("No images to predict. Check your test data.")