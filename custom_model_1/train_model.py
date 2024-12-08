import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tqdm import tqdm
import joblib  # To save the LabelEncoder object

# Function to load images and their corresponding labels
def load_images_and_labels(data_path, ascii_path, limit=None):
    """
    Loads images and labels from the dataset.

    Args:
        data_path (str): Path to the directory containing images.
        ascii_path (str): Path to the ASCII file containing labels.
        limit (int, optional): Maximum number of images to load. If None, loads all available images.

    Returns:
        images (list): List of loaded images as numpy arrays.
        labels (list): List of corresponding labels.
    """
    images = []
    labels = []

    # Read the 'words.txt' file containing label information
    ascii_file = os.path.join(ascii_path, "words.txt")
    with open(ascii_file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Loading images"):  # Display progress bar for image loading
        if line.startswith("#"):  # Skip comment lines
            continue

        # Split the line into parts and ensure it contains valid data
        parts = line.strip().split(" ")
        if len(parts) < 9 or parts[1] != "ok":  # Skip invalid lines
            continue

        image_name = parts[0]  # Extract the image name
        label = parts[-1]  # Extract the label (last part of the line)

        # Construct the path to the image file by searching through subdirectories
        image_path = None
        for subdir, _, files in os.walk(data_path):  # Walk through subdirectories
            if image_name + ".png" in files:
                image_path = os.path.join(subdir, image_name + ".png")
                break

        # Skip if the image file is not found
        if not image_path:
            print(f"Image not found: {image_name}.png")
            continue

        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue

        # Add the image and label to the lists
        images.append(img)
        labels.append(label)

        # Break if the specified limit is reached
        if limit and len(images) >= limit:
            break

    return images, labels

# The dataset can be found in https://fki.tic.heia-fr.ch/databases/iam-handwriting-database. "words.txt" inside ascii.tgz and the images of the words inside words.tgz is used.
# Define paths to the dataset and ASCII files
DATA_PATH = "dataset/words"
ASCII_PATH = "dataset/ascii"
LIMIT = 100000  # Set a limit on the number of images (use None for all data)

# Load images and labels
images, labels = load_images_and_labels(DATA_PATH, ASCII_PATH, limit=None)

# Filter out invalid images and their labels
valid_images = []
valid_labels = []
for img, label in zip(images, labels):
    if img is not None:
        valid_images.append(img)
        valid_labels.append(label)
    else:
        print("Invalid image skipped.")

# Resize images to a fixed size (128x32) and normalize pixel values
try:
    images_resized = [cv2.resize(img, (128, 32)) for img in valid_images]
except cv2.error as e:
    print(f"Resize error: {e}")
    exit()

# Normalize pixel values to the range [0, 1] and reshape for the model input
images_normalized = np.array(images_resized) / 255.0
images_normalized = images_normalized.reshape(-1, 32, 128, 1)  # Add a channel dimension

# Encode the labels into numeric format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(valid_labels)
encoded_labels = tf.keras.utils.to_categorical(encoded_labels)  # Convert to one-hot encoding

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_normalized, encoded_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.InputLayer(input_shape=(32, 128, 1)),  # Input layer for grayscale images of size 128x32
    layers.Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Flatten(),  # Flatten the output for the dense layers
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax activation
])

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data and validate on the validation set
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), verbose=1)

# Save the LabelEncoder for future use
joblib.dump(label_encoder, "label_encoder.pkl")  # Save the LabelEncoder to a file

# Save the trained model to an HDF5 file
model.save('handwriting_recognition_model.h5')
print("Model successfully saved!")
