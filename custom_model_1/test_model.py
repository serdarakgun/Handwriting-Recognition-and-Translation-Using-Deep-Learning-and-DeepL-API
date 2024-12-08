import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd  # To read the Excel file
from tqdm import tqdm  # For progress tracking

# Load the pre-trained model and LabelEncoder
model = tf.keras.models.load_model('handwriting_recognition_model.h5')
# Load the saved handwriting recognition model.
label_encoder = joblib.load("label_encoder.pkl")
# Load the saved LabelEncoder object to decode class labels.

# Function to read the Excel file containing image paths and labels
def load_labels_from_excel(file_path):
    """
    Reads an Excel (CSV format) file containing image paths and their labels.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with 'Path' and 'Label' columns if successful, None otherwise.
    """
    try:
        data = pd.read_csv(file_path, header=None, names=["Path", "Label"])
        # Read the CSV file, assuming no headers and assign custom column names.
        return data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# Function to load test images and labels based on the DataFrame
def load_test_images_from_excel(data_frame):
    """
    Loads test images and their corresponding labels from the given DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame containing 'Path' and 'Label'.

    Returns:
        tuple: (images, labels, file_names), where:
               - images: List of loaded image arrays.
               - labels: List of true labels.
               - file_names: List of file paths.
    """
    images = []
    labels = []
    file_names = []

    for _, row in data_frame.iterrows():  # Iterate over each row in the DataFrame
        image_path = row["Path"]
        label = row["Label"]

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            # Load the image in grayscale mode
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Ensure the image was loaded successfully
                images.append(img)
                labels.append(label)
                file_names.append(image_path)
            else:
                print(f"Could not read image: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return images, labels, file_names

# Path to the Excel file containing test image paths and labels
EXCEL_FILE_PATH = "preprocessed.csv"  # Update this to the correct file path

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
        resized_img = cv2.resize(img, (128, 32))  # Resize to (128x32)
        test_images_resized.append(resized_img)
    except Exception as e:
        print(f"Error resizing image: {e}")

if len(test_images_resized) == 0:
    raise ValueError("No valid test images found. Please check the input folder and image format.")

# Normalize image pixel values to the range [0, 1]
test_images_normalized = np.array(test_images_resized) / 255.0
test_images_normalized = test_images_normalized.reshape(-1, 32, 128, 1)
# Reshape to include the channel dimension for grayscale images.

# Perform predictions if there are images to predict
if test_images_normalized.size > 0:
    predictions = model.predict(test_images_normalized)
    # Get predicted probabilities for each class.
    predicted_labels = np.argmax(predictions, axis=1)
    # Convert probabilities to class indices.
    decoded_labels = label_encoder.inverse_transform(predicted_labels)
    # Decode the predicted indices to the original class labels.

    # Count matches and mismatches
    match_count = 0
    mismatch_count = 0

    for real_label, predicted_label in zip(real_labels, decoded_labels):
        if real_label == predicted_label:  # Check if prediction matches the true label
            match_count += 1
        else:
            mismatch_count += 1

    # Optional: Visualize predictions with corresponding labels (commented for larger datasets)
    '''
    for i, (img, prediction, real_label, file_name) in enumerate(zip(test_images, decoded_labels, real_labels, file_names)):
        plt.imshow(img, cmap='gray')
        plt.title(f"File: {file_name}\nReal: {real_label}\nPrediction: {prediction}")
        plt.axis('off')
        plt.show()
    '''

    # Print match and mismatch counts
    print(f"Total Matches: {match_count}")
    print(f"Total Mismatches: {mismatch_count}")
else:
    print("No images to predict. Check your test data.")
