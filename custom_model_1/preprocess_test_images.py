import cv2
import os
from glob import glob


def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read all image files in the input folder
    image_paths = glob(os.path.join(input_folder, '*'))

    for image_path in image_paths:
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Apply Gaussian Blur to reduce high-frequency noise
        blurred = cv2.GaussianBlur(img, (7, 7), 0)

        # Apply adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 3)

        # Perform morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Remove small specks using contour filtering (optional)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = morph.copy()
        for contour in contours:
            if cv2.contourArea(contour) < 10:  # Filter out small contours (adjust size as needed)
                cv2.drawContours(cleaned, [contour], -1, 0, -1)

        # Save the processed image
        file_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, cleaned)
        print(f"Processed and saved: {output_path}")


# Input and output folder paths
input_folder = 'test_images'  # Replace with your input folder path
output_folder = 'processed_test_images'  # Replace with your desired output folder path

# Process images
process_images(input_folder, output_folder)
